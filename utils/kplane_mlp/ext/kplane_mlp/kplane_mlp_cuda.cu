#include <torch/extension.h>
// #include <ATen/OpMathType.h>
// #include <ATen/native/cuda/GridSampler.h>
// #include <ATen/native/GridSamplerUtils.h>
#include <ATen/native/cuda/GridSampler.cuh>
#include <ATen/native/cuda/UpSample.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/native/cuda/KernelUtils.cuh>
// #include <ATen/core/TensorBase.h>
#include <ATen/Dispatch.h>
#include <c10/macros/Macros.h>
#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

// WMMA (Tensor Core)
#include <mma.h>
using namespace nvcuda;

namespace at::native {

using namespace at::cuda::detail;

using at::native::detail::GridSamplerInterpolation;
using at::native::detail::GridSamplerPadding;

// MODIFIED BASED ON: https://github.com/pytorch/pytorch/blob/31bb65de195d5c3cc5d74e7b909495ef8c2035e7/aten/src/ATen/native/cuda/GridSampler.cu

// #if TORCH_VERSION_MAJOR > 1
// CONST_TI(st, it, ten) getTensorInfo<const st, it>(ten)
// #else
// CONST_TI(st, it, ten) getTensorInfo<st, it>(ten)
// #endif

// after torch 2.3 use const
#if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 3
#define USE_CONST_TI 1
// CONST_TI(st, it, ten) (getTensorInfo<const st, it>(ten))
// TI(st, it, ten)       (getTensorInfo<st, it>(ten))
#else
#define USE_CONST_TI 0
// CONST_TI(st, it, ten) (getTensorInfo<st, it>(ten))
// TI(st, it, ten)       (getTensorInfo<st, it>(ten))
#endif

#define WARP_CHUNK_K 8

// #define WMMA_M 8
// #define WMMA_N 32
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// #define SHMEM_WMMA_HALF    1
#define SHMEM_WMMA_BUDGET 0

namespace {

  enum FeatFusion{SUM=0, AVG=1, MUL=2};
  enum Activation{Softplus=0, ReLU=1};

  // template <typename scalar_t, typename index_t>
  // __device__ void mlp_fn(
  //   scalar_t* output, const index_t out_s, 
  //   const scalar_t* vector, const index_t vec_s, 
  //   const scalar_t* matrix, const index_t mat_s,
  //   const index_t ci, const index_t co
  // ){
  //   for (index_t i=0; i<co; ++i){
  //     scalar_t out = static_cast<scalar_t>(0);
  //     for (index_t j=0; j<ci; ++j){
  //       out += vector[j*vec_s] * matrix[(i*ci+j)*mat_s];
  //     }
  //     output[i*out_s] = out + matrix[(co*ci+i)*mat_s]; // bias
  //   }
  // }

  template <typename scalar_t, typename index_t>
  __inline__ __device__ void slow_warp_fc(
    const float*    mat_a,  //  8 x ci row_major
    const scalar_t* mat_b,  // ci x co col_major
          float*    mat_c,  //  8 x co row_major
    
    const index_t rs_a, const index_t cs_b, const index_t rs_c,

    const index_t ci, const index_t co
  ){
    const index_t tile_K = 4;  // h
    const index_t tile_I = 8;  // w

    const index_t tid = threadIdx.x;
    // const index_t item_i = threadIdx.x / tile_K;

    for (index_t m=tid/tile_K; m<WARP_CHUNK_K; m+=tile_K){ // [0, 0, 0, 0, 1, 1, 1, 1, ...]
      // for (index_t o=threadIdx.x; o<co; ++o){
      for (index_t o=tid%tile_I; o<co; o+=tile_I){         // [0, 1, 2, 3, 4, 5, 6, 7]

        // half out = __int2half_rn(0);
        // #pragma unroll
        // for (index_t i=0; i<ci; ++i){
        //   out = __hfma(mat_a[m*rs_a + i], mat_b[i + o*cs_b], out); // mat_a[m,i] * mat_b[i,o]
        // }
        // mat_c[m*rs_c + o] = __half2float(out);  // mat_c[m,o] = 

        scalar_t out = static_cast<scalar_t>(0);
        #pragma unroll
        for (index_t i=0; i<ci; ++i){
          out += mat_a[m*rs_a + i] * __half2float(mat_b[i + o*cs_b]); // mat_a[m,i] * mat_b[i,o]
        }
        mat_c[m*rs_c + o] = out;  // mat_c[m,o] = 

      }
    }
  }

  template <typename scalar_t, typename index_t>
  __inline__ __device__ void wmma_warp_fc(
    const half*     mat_a,   //  8 x ci row_major stride=64
    const half*     mat_b,   // ci x co col_major stride=64
          scalar_t* mat_c,   //  8 x co row_major stride=64

    const index_t rs_a, const index_t cs_b, const index_t rs_c,
    // index_t rs_a, index_t cs_b, index_t rs_c,

    const index_t ci, const index_t co
  ){
      wmma::fragment<wmma::matrix_a,    WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
      wmma::fragment<wmma::matrix_b,    WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
      wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>                 c_frag;
      // wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>                 d_frag;

      // Initialize the output to zero
      wmma::fill_fragment(c_frag, 0.0f); // bias

      for (int o=0; o<co; o+=WMMA_K){
        scalar_t* c_ptr = mat_c + o*64;
        for (int i=0; i<ci; i+=WMMA_N){

          const __restrict__ half* a_ptr = mat_a + i          ;
          const __restrict__ half* b_ptr = mat_b + o*cs_b + i ;

          // Load the inputs
          wmma::load_matrix_sync(a_frag, a_ptr, rs_a);
          wmma::load_matrix_sync(b_frag, b_ptr, cs_b);

          // Perform the matrix multiplication
          wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

          // Store the output
        }
        wmma::store_matrix_sync(c_ptr, c_frag, rs_c, wmma::mem_row_major);
      }
  }

  template <typename scalar_t, typename index_t>
  __inline__ __device__ void act_fn(
    scalar_t* output, const index_t out_s,
    const index_t c,
    const Activation act
  ){
    if (act == Activation::Softplus){
      float beta      = static_cast<float>(10); // same as tiny-cuda-nn https://github.com/NVlabs/tiny-cuda-nn/blob/b3473c81396fe927293bdfd5a6be32df8769927c/include/tiny-cuda-nn/common_device.h#L100
      float threshold = static_cast<float>(20);
      for (int i=0; i<c; ++i){
        scalar_t x   = output[i*out_s];
        float beta_x = beta * ((float)x);
        output[i*out_s] = beta_x > threshold ? x : (scalar_t)(log1pf(expf(beta_x))/beta);
      }
    } else if (act == Activation::ReLU){
      for (int i=0; i<c; ++i){
        output[i*out_s] = max(output[i*out_s], static_cast<scalar_t>(0));
      }
    }
  }

  template <typename scalar_t, typename index_t>
  __launch_bounds__(32)
  __global__ void kplane_mlp_forward_kernel(
      const index_t nthreads,
      const index_t num_kpl_skew,
      const index_t num_out_skew,

      #if USE_CONST_TI > 0
      TensorInfo<const scalar_t, index_t> kpl_param, // B, H, W, K, C
      TensorInfo<const scalar_t, index_t> mlp_param, // B, [L0, L1, ...]
      TensorInfo<const scalar_t, index_t> grid,      // B, p, q, K, 2
      TensorInfo<const scalar_t, index_t> skip,      // B, p, q, sC
      #else
      TensorInfo<scalar_t, index_t> kpl_param, // B, H, W, K, C
      TensorInfo<scalar_t, index_t> mlp_param, // B, [L0, L1, ...]
      TensorInfo<scalar_t, index_t> grid,      // B, p, q, K, 2
      TensorInfo<scalar_t, index_t> skip,      // B, p, q, sC
      #endif
      TensorInfo<scalar_t, index_t> output,          // B, p, q, oC

      const GridSamplerInterpolation interpolation_mode,
      const GridSamplerPadding padding_mode,
      const bool align_corners, 
      const FeatFusion feat_fusion,
      
      const index_t mlp_layers, 
      const index_t mlp_dim_hidden, 
      const index_t mlp_dim_output, 
      const Activation activation) {

    // using opmath_t = at::opmath_type<scalar_t>;
    const index_t inp_H  = kpl_param.sizes[1];
    const index_t inp_W  = kpl_param.sizes[2];
    const index_t inp_K  = kpl_param.sizes[3];
    const index_t inp_C  = kpl_param.sizes[4];
    const index_t inp_sN = kpl_param.strides[0];
    const index_t inp_sH = kpl_param.strides[1];
    const index_t inp_sW = kpl_param.strides[2];
    const index_t inp_sK = kpl_param.strides[3];
    const index_t inp_sC = kpl_param.strides[4];

    const index_t mlp_sN  = mlp_param.strides[0];
    const index_t mlp_sP  = mlp_param.strides[1];

    const index_t out_N      = grid.sizes[0];
    const index_t out_H      = grid.sizes[1];
    const index_t out_W      = grid.sizes[2];
    const index_t grid_sN    = grid.strides[0];
    const index_t grid_sH    = grid.strides[1];
    const index_t grid_sW    = grid.strides[2];
    const index_t grid_sK    = grid.strides[3];
    const index_t grid_sCoor = grid.strides[4];

    const index_t skip_C  = skip.sizes[3];
    const index_t skip_sN = skip.strides[0];
    const index_t skip_sH = skip.strides[1];
    const index_t skip_sW = skip.strides[2];
    const index_t skip_sC = skip.strides[3];

    const index_t out_C  = output.sizes[3];
    const index_t out_sN = output.strides[0];
    const index_t out_sH = output.strides[1];
    const index_t out_sW = output.strides[2];
    const index_t out_sC = output.strides[3];

    extern __shared__ char shmem[];
    // scalar shmem_kpl_ptr[WARP_CHUNK_K][inp_C + skip_C]
    // scalar shmem_out_ptr[WARP_CHUNK_K][max(kpl_C, mid_C, out_C)]
    // half  shmem_mata_ptr[WARP_CHUNK_K][64]
    // half  shmem_matb_ptr[64][64]

    const index_t tid  = threadIdx.x;
    const index_t n    = blockIdx.y;

    const int32_t kpl_base  = 0;
    const int32_t out_base  = sizeof(scalar_t)*WARP_CHUNK_K*(inp_C+skip_C) + num_kpl_skew;
    const int32_t mata_base = out_base  + num_out_skew;
    const int32_t matb_base = matb_base + sizeof(half)*WARP_CHUNK_K*64;

    float* shmem_kpl_ptr = (float*)(shmem);            // scalar shmem_kpl_ptr[WARP_CHUNK_K][inp_C + skip_C]             # 8*(<64) = <0.5K
    float* shmem_out_ptr = (float*)(shmem+out_base);   // scalar shmem_out_ptr[WARP_CHUNK_K][max(kpl_C, mid_C, out_C)]   # 8*(<64) = <0.5K

    // scalar_t*    shmem_out_ptr = (scalar_t*)(shmem+out_base);   // scalar shmem_out_ptr[WARP_CHUNK_K][max(out_C, mlp_hidden_dim)]   # 8*(<64) = <0.5K

    // #if SHMEM_WMMA_HALF > 0
    half* shmem_mata_ptr = (half*)(shmem+mata_base);         // scalar shmem_mata_ptr[WARP_CHUNK_K][64]  # vector  8*64  = 0.5K
    half* shmem_matb_ptr = (half*)(shmem+matb_base);         // scalar shmem_matb_ptr[64][64]            # weight  64*64 = 4K
    // #else
    // scalar_t* shmem_mata_ptr = (scalar_t*)(shmem+mata_base);         // scalar shmem_mata_ptr[WARP_CHUNK_K][64]  # vector  8*64  = 0.5K
    // scalar_t* shmem_matb_ptr = (scalar_t*)(shmem+matb_base);         // scalar shmem_matb_ptr[64][64]            # weight  64*64 = 4K
    // #endif

    const int32_t item_C = (inp_C + skip_C);

    const __restrict__ scalar_t* kpl_data  = kpl_param.data;
    const __restrict__ scalar_t* mlp_data  = mlp_param.data;
    const __restrict__ scalar_t* skip_data = skip.data;
    const __restrict__ scalar_t* grid_data = grid.data;

    // init
    {
      float init_value = static_cast<float>(0);
      if (feat_fusion == FeatFusion::MUL)
        init_value = static_cast<float>(1);

      for (index_t ci=tid; ci<WARP_CHUNK_K*(inp_C+skip_C); ci+=nthreads)
        shmem_kpl_ptr[ci] = init_value;
    }

    for (int32_t item_i=0; item_i<WARP_CHUNK_K; ++item_i){
      const index_t index = blockIdx.x*WARP_CHUNK_K + item_i;

      if (index > out_W*out_H)
        break;

      float* buff_s = shmem_kpl_ptr + item_i*item_C;

      const index_t w = index % out_W;
      const index_t h = (index / out_W) % out_H;
      const index_t grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;

      // concatenate
      {
        auto skip_offset = n*skip_sN + h*skip_sH + w*skip_sW;
        for (index_t i=tid; i<skip_C; i+=nthreads)
          buff_s[inp_C+i] = skip_data[skip_offset + i*skip_sC];
      }
      
      // __sync_threads();

      const __restrict__ scalar_t* grid_ptr_NHW = grid_data + grid_offset;
      // k-plane feature query
      for (index_t ki=0; ki<inp_K; ++ki){
        // get the corresponding input x, y co-ordinates from grid
        // scalar_t x = grid_data[grid_offset + ki*grid_sK];
        // scalar_t y = grid_data[grid_offset + ki*grid_sK + grid_sCoor];
        scalar_t x = grid_ptr_NHW[ki*grid_sK];
        scalar_t y = grid_ptr_NHW[ki*grid_sK + grid_sCoor];

        scalar_t ix = grid_sampler_compute_source_index(x, inp_W, padding_mode, align_corners);
        scalar_t iy = grid_sampler_compute_source_index(y, inp_H, padding_mode, align_corners);

        if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
          // get NE, NW, SE, SW pixel values from (x, y)
          index_t ix_nw = static_cast<int32_t>(::floor(ix));
          index_t iy_nw = static_cast<int32_t>(::floor(iy));
          index_t ix_ne = ix_nw + 1;
          index_t iy_ne = iy_nw;
          index_t ix_sw = ix_nw;
          index_t iy_sw = iy_nw + 1;
          index_t ix_se = ix_nw + 1;
          index_t iy_se = iy_nw + 1;

          // get surfaces to each neighbor:
          scalar_t nw = (ix_se - ix)    * (iy_se - iy);
          scalar_t ne = (ix    - ix_sw) * (iy_sw - iy);
          scalar_t sw = (ix_ne - ix)    * (iy    - iy_ne);
          scalar_t se = (ix    - ix_nw) * (iy    - iy_nw);

          const bool ib_nw = within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W);
          const bool ib_ne = within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W);
          const bool ib_sw = within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W);
          const bool ib_se = within_bounds_2d(iy_se, ix_se, inp_H, inp_W);

          index_t ix_nw_safe = ib_nw ? ix_nw : ix_nw;
          index_t ix_ne_safe = ib_ne ? ix_ne : ix_nw;
          index_t ix_sw_safe = ib_sw ? ix_sw : ix_nw;
          index_t ix_se_safe = ib_se ? ix_se : ix_nw;
          index_t iy_nw_safe = ib_nw ? iy_nw : iy_nw;
          index_t iy_ne_safe = ib_ne ? iy_ne : iy_nw;
          index_t iy_sw_safe = ib_sw ? iy_sw : iy_nw;
          index_t iy_se_safe = ib_se ? iy_se : iy_nw;

          nw *= ib_nw;
          ne *= ib_ne;
          sw *= ib_sw;
          se *= ib_se;

          const __restrict__ scalar_t* inp_ptr_NC = kpl_param.data + n * inp_sN + ki * inp_sK;

          for (index_t ci=tid; ci<inp_C; ci+=nthreads){
            // calculate bilinear weighted pixel value and set output pixel

            const __restrict__ scalar_t* inp_ptr_NC2 = inp_ptr_NC + ci * inp_sC;

            scalar_t cv = static_cast<scalar_t>(0);
            cv += inp_ptr_NC2[iy_nw_safe * inp_sH + ix_nw_safe * inp_sW] * nw;
            cv += inp_ptr_NC2[iy_ne_safe * inp_sH + ix_ne_safe * inp_sW] * ne;
            cv += inp_ptr_NC2[iy_sw_safe * inp_sH + ix_sw_safe * inp_sW] * sw;
            cv += inp_ptr_NC2[iy_se_safe * inp_sH + ix_se_safe * inp_sW] * se;

            if (feat_fusion == FeatFusion::SUM)
              buff_s[ci] += cv;
            else if (feat_fusion == FeatFusion::AVG)
              buff_s[ci] += cv/inp_K;
            else if (feat_fusion == FeatFusion::MUL)
              buff_s[ci] *= cv;
          }
          // __sync_threads();
        
        } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
          index_t ix_nearest = static_cast<index_t>(std::nearbyint(ix));
          index_t iy_nearest = static_cast<index_t>(std::nearbyint(iy));

          const bool ib = within_bounds_2d(iy_nearest, ix_nearest, inp_H, inp_W);

          for (index_t ci=tid; ci<inp_C; ci+=nthreads){
            const __restrict__ scalar_t* inp_ptr_NC = kpl_param.data + n * inp_sN + ki * inp_sK + ci * inp_sC;

            scalar_t cv = static_cast<scalar_t>(0);
            if (ib)
              cv = inp_ptr_NC[iy_nearest * inp_sH + ix_nearest * inp_sW];

            if (feat_fusion == FeatFusion::SUM)
              buff_s[ci] += cv;
            else if (feat_fusion == FeatFusion::AVG)
              buff_s[ci] += cv/inp_K;
            else if (feat_fusion == FeatFusion::MUL)
              buff_s[ci] *= cv;
          }
          // __sync_threads();

        }
      }

    }

    if (mlp_layers == 0){
      // save
      for (index_t i=tid; i<WARP_CHUNK_K*out_C; i+=nthreads)
        shmem_out_ptr[i] = shmem_kpl_ptr[i];
    } else {
      // MLP

      // index_t ci = inp_C + skip_C, co = mlp_dim_hidden;

      // // auto matrix = mlp_param.data + n*mlp_sN;

      // index_t mlp_base = 0;

      // half* mata_ptr = shmem_mata_ptr;
      // half* matb_ptr = shmem_matb_ptr;
      // float* matc_ptr = (float*)shmem_out_ptr;

      // // for (index_t i=tid; i<WARP_CHUNK_K*ci; i+=nthreads)
      // //   mata_ptr[i] = __float2half(shmem_kpl_ptr[i]);

      // for (index_t li=0; li<mlp_layers; ++li){

      //   // load weight
      //   for (index_t i=tid; i<64*64; i+=nthreads)
      //     matb_ptr[i] = __int2half_rn(0);
      //   for (index_t i=0; i<co; ++i)
      //     for (index_t j=tid; j<ci; j+=nthreads)
      //       matb_ptr[i*64+j] = __float2half(mlp_data[n*mlp_sN + mlp_base + i*ci + j]);  // W_li[B][j][i]
        
      //   if (li == 0){
      //     // slow_warp_fc<float, index_t>(mata_ptr, matb_ptr, matc_ptr, ci, ci, 64, ci, co);
      //     slow_warp_fc<float, index_t>(shmem_kpl_ptr, matb_ptr, matc_ptr, ci, ci, 64, ci, co);

      //   }else{

      //   // #if SHMEM_WMMA_HALF > 0
      //     wmma_warp_fc<float, unsigned>(mata_ptr, matb_ptr, matc_ptr, 64, 64, 64, ci, co);
      //   // #else
      //   //   wmma_warp_fc<float, unsigned>(mata_ptr, matb_ptr, matc_ptr, 64, 64, 64, ci, co);
      //   // #endif

      //   }

      //   // if (li < mlp_layers-1){
      //   //   act_fn<float, index_t>(matc_ptr, 1, WARP_CHUNK_K*co, activation);
      //   //   // act_fn<float, index_t>(matc_ptr, 1, shmem_out_ptr, activation);

      //   // // #if SHMEM_WMMA_HALF > 0
      //   //   for (index_t i=tid; i<WARP_CHUNK_K*co; i+=nthreads)
      //   //     mata_ptr[i] = __float2half(matc_ptr[i]);
      //   // // #else
      //   // //   for (index_t i=tid; i<WARP_CHUNK_K*co; i+=nthreads)
      //   // //     mata_ptr[i] = __float2half(matc_ptr[i]);
      //   // // #endif

      //   // } else {
      //     for (index_t i=tid; i<WARP_CHUNK_K*co; i+=nthreads)
      //       shmem_out_ptr[i] = matc_ptr[i];
      //   // }

      //   // else
      //   //   act_fn<scalar_t, index_t>(out, 1, co, last_activation);

      //   // scalar_t* temp_ = matc_ptr;
      //   // matc_ptr = mata_ptr;
      //   // mata_ptr = temp_;

      //   // matrix += (co*(ci+1))*mlp_sP;
      //   ci = co;

      //   if (li == mlp_layers - 1)
      //     co = mlp_dim_output;

      //   // if (threadIdx.x == 0)
      //   //   printf("%d")
      // }

      // // save
      // auto output_offset = n*out_sN + h*out_sH + w*out_sW;
      // for (index_t i=tid; i<out_C; i+=nthreads){
      //   output.data[output_offset + i*out_sC] = out[i];
      // }
    }


    for (int32_t item_i=0; item_i<WARP_CHUNK_K; ++item_i){
      const index_t index = blockIdx.x*WARP_CHUNK_K + item_i;

      if (index > out_W*out_H)
        return;

      const float* buff_o = shmem_out_ptr + item_i*out_C;

      const index_t w = index % out_W;
      const index_t h = (index / out_W) % out_H;

      auto output_offset = n*out_sN + h*out_sH + w*out_sW;
      for (index_t i=tid; i<out_C; i+=nthreads){
        output.data[output_offset + i*out_sC] = buff_o[i];
      }
    }
  }

  // Note [Passing pointer and offset to fastAtomicAdd]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // For its internal bounds checking, fastAtomicAdd needs to know where the destination address
  // lies relative to the entire tensor, so we pass the base grad_input.data and full offset information,
  // including batch * channel offset (NC_offset).

  template <typename scalar_t, typename index_t>
  __launch_bounds__(32)
  __global__ void kplane_mlp_backward_kernel(
      const index_t nthreads,
      #if USE_CONST_TI > 0
      TensorInfo<const scalar_t, index_t> grad_output,

      TensorInfo<const scalar_t, index_t> kpl_param, // B, H, W, K, C
      TensorInfo<const scalar_t, index_t> mlp_param, // B, [L0, L1, ...]
      TensorInfo<const scalar_t, index_t> grid,      // B, p, q, K, 2
      TensorInfo<const scalar_t, index_t> skip,      // B, p, q, sC
      #else
      TensorInfo<scalar_t, index_t> grad_output,

      TensorInfo<scalar_t, index_t> kpl_param, // B, H, W, K, C
      TensorInfo<scalar_t, index_t> mlp_param, // B, [L0, L1, ...]
      TensorInfo<scalar_t, index_t> grid,      // B, p, q, K, 2
      TensorInfo<scalar_t, index_t> skip,      // B, p, q, sC
      #endif

      TensorInfo<scalar_t, index_t> grad_kpl,    // B, H, W, K, C  initialized to zeros (or unused if input_requires_grad is false)
      TensorInfo<scalar_t, index_t> grad_mlp,    // initialized to empty
      TensorInfo<scalar_t, index_t> grad_grid,   // B, p, q, K, 2 initialized to empty (0 if interpolation == "nearest")
      TensorInfo<scalar_t, index_t> grad_skip,   // B, p, q, sC   initialized to empty

      const GridSamplerInterpolation interpolation_mode,
      const GridSamplerPadding padding_mode,
      const bool align_corners, 
      const FeatFusion feat_fusion,
      
      const index_t mlp_layers, 
      const index_t mlp_dim_hidden, 
      const index_t mlp_dim_output, 
      const Activation activation,

      const index_t grad_kpl_memory_span,
      const index_t grad_mlp_memory_span,
      const index_t grad_grid_memory_span,
      const index_t grad_skip_memory_span
      ) {
      
    const bool kpl_requires_grad  = grad_kpl_memory_span  > 0;
    const bool mlp_requires_grad  = grad_mlp_memory_span  > 0;
    const bool grid_requires_grad = grad_grid_memory_span > 0;
    const bool skip_requires_grad = grad_skip_memory_span > 0;

    // using opmath_t = at::opmath_type<scalar_t>;
    const index_t inp_H  = kpl_param.sizes[1];
    const index_t inp_W  = kpl_param.sizes[2];
    const index_t inp_K  = kpl_param.sizes[3];
    const index_t inp_C  = kpl_param.sizes[4];
    const index_t inp_sN = kpl_param.strides[0];
    const index_t inp_sH = kpl_param.strides[1];
    const index_t inp_sW = kpl_param.strides[2];
    const index_t inp_sK = kpl_param.strides[3];
    const index_t inp_sC = kpl_param.strides[4];

    const index_t mlp_sN  = mlp_param.strides[0];
    const index_t mlp_sP  = mlp_param.strides[1];

    const index_t out_H      = grid.sizes[1];
    const index_t out_W      = grid.sizes[2];

    const index_t grid_sN    = grid.strides[0];
    const index_t grid_sH    = grid.strides[1];
    const index_t grid_sW    = grid.strides[2];
    const index_t grid_sK    = grid.strides[3];
    const index_t grid_sCoor = grid.strides[4];

    const index_t skip_C  = skip.sizes[3];
    const index_t skip_sN = skip.strides[0];
    const index_t skip_sH = skip.strides[1];
    const index_t skip_sW = skip.strides[2];
    const index_t skip_sC = skip.strides[3];

    // const index_t out_sN = output.strides[0];
    // const index_t out_sH = output.strides[1];
    // const index_t out_sW = output.strides[2];
    // const index_t out_sC = output.strides[3];

    extern __shared__ char shmem[];
    scalar_t* buff_s = (scalar_t*)shmem;

    const index_t gOut_sN = grad_output.strides[0];
    const index_t gOut_sH = grad_output.strides[1];
    const index_t gOut_sW = grad_output.strides[2];
    const index_t gOut_sC = grad_output.strides[3];

    index_t gKpl_sN;
    index_t gKpl_sH;
    index_t gKpl_sW;
    index_t gKpl_sK;
    index_t gKpl_sC;

    index_t gGrid_sN;
    index_t gGrid_sH;
    index_t gGrid_sW;
    index_t gGrid_sK;
    index_t gGrid_sCoor;

    index_t gSkip_sN;
    index_t gSkip_sH;
    index_t gSkip_sW;
    index_t gSkip_sC;

    if (kpl_requires_grad) {
      gKpl_sN = grad_kpl.strides[0];
      gKpl_sH = grad_kpl.strides[1];
      gKpl_sW = grad_kpl.strides[2];
      gKpl_sK = grad_kpl.strides[3];
      gKpl_sC = grad_kpl.strides[4];
    }
    if (grid_requires_grad) {
      gGrid_sN = grad_grid.strides[0];
      gGrid_sH = grad_grid.strides[1];
      gGrid_sW = grad_grid.strides[2];
      gGrid_sK = grad_grid.strides[3];
      gGrid_sCoor  = grad_grid.strides[4];
    }
    if (skip_requires_grad) {
      gSkip_sN = grad_skip.strides[0];
      gSkip_sH = grad_skip.strides[1];
      gSkip_sW = grad_skip.strides[2];
      gSkip_sC = grad_skip.strides[3];
    }
    // index_t gGrid_sW = grad_grid.strides[2];

    {
      const index_t index = blockIdx.x;
      const index_t tid   = threadIdx.x;

      const index_t w = index % out_W;
      const index_t h = (index / out_W) % out_H;
      const index_t n = index / (out_H * out_W);
      const auto grid_offset  = n * grid_sN  + h * grid_sH  + w * grid_sW;
      const auto gOut_offset  = n * gOut_sN  + h * gOut_sH  + w * gOut_sW;
      const auto gGrid_offset = n * gGrid_sN + h * gGrid_sH + w * gGrid_sW;
      const auto gSkip_offset = n * gSkip_sN + h * gSkip_sH + w * gSkip_sW;

      // init buff_s[K][inp_C] = grad[bi,hi,wi,:].expand(K,-1)
      {
        for (index_t ci=tid; ci<inp_C; ci+=nthreads){
          scalar_t gOut_val = grad_output.data[gOut_offset + (ci)*gOut_sC];
          for (index_t ki=0; ki<inp_K; ++ki)
            buff_s[ki*inp_C+ci] = gOut_val;
        }
      }

      // concatenate
      if (skip_requires_grad) {
        for (index_t i=tid; i<skip_C; i+=nthreads)
          grad_skip.data[gSkip_offset + i*gSkip_sC] = grad_output.data[gOut_offset + (inp_C+i)*gOut_sC];
      }
      
      // __sync_threads();

      // set input correlated gradient
      if (feat_fusion == FeatFusion::SUM) {
      }
      else if (feat_fusion == FeatFusion::AVG){
        for (index_t ci=tid; ci<inp_K*inp_C; ci+=nthreads)
          buff_s[ci] /= inp_K;
      }
      else if (feat_fusion == FeatFusion::MUL){
      for (index_t ki=0; ki<inp_K; ++ki){
        // get the corresponding input x, y co-ordinates from grid
        scalar_t x = grid.data[grid_offset + ki*grid_sK];
        scalar_t y = grid.data[grid_offset + ki*grid_sK + grid_sCoor];

        scalar_t ix = grid_sampler_compute_source_index(x, inp_W, padding_mode, align_corners);
        scalar_t iy = grid_sampler_compute_source_index(y, inp_H, padding_mode, align_corners);

        if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
          // get NE, NW, SE, SW pixel values from (x, y)
          index_t ix_nw = static_cast<index_t>(::floor(ix));
          index_t iy_nw = static_cast<index_t>(::floor(iy));
          index_t ix_ne = ix_nw + 1;
          index_t iy_ne = iy_nw;
          index_t ix_sw = ix_nw;
          index_t iy_sw = iy_nw + 1;
          index_t ix_se = ix_nw + 1;
          index_t iy_se = iy_nw + 1;

          // get surfaces to each neighbor:
          scalar_t nw = (ix_se - ix)    * (iy_se - iy);
          scalar_t ne = (ix    - ix_sw) * (iy_sw - iy);
          scalar_t sw = (ix_ne - ix)    * (iy    - iy_ne);
          scalar_t se = (ix    - ix_nw) * (iy    - iy_nw);

          const bool ib_nw = within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W);
          const bool ib_ne = within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W);
          const bool ib_sw = within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W);
          const bool ib_se = within_bounds_2d(iy_se, ix_se, inp_H, inp_W);

          index_t ix_nw_safe = ib_nw ? ix_nw : ix_nw;
          index_t ix_ne_safe = ib_ne ? ix_ne : ix_nw;
          index_t ix_sw_safe = ib_sw ? ix_sw : ix_nw;
          index_t ix_se_safe = ib_se ? ix_se : ix_nw;
          index_t iy_nw_safe = ib_nw ? iy_nw : iy_nw;
          index_t iy_ne_safe = ib_ne ? iy_ne : iy_nw;
          index_t iy_sw_safe = ib_sw ? iy_sw : iy_nw;
          index_t iy_se_safe = ib_se ? iy_se : iy_nw;

          nw *= ib_nw;
          ne *= ib_ne;
          sw *= ib_sw;
          se *= ib_se;

          auto inp_ptr_NC = kpl_param.data + n * inp_sN + ki * inp_sK;

          for (index_t ci=tid; ci<inp_C; ci+=nthreads){
            // calculate bilinear weighted pixel value and set output pixel

            auto inp_ptr_NC2 = inp_ptr_NC + ci * inp_sC;

            scalar_t cv = static_cast<scalar_t>(0);
            cv += inp_ptr_NC2[iy_nw_safe * inp_sH + ix_nw_safe * inp_sW] * nw;
            cv += inp_ptr_NC2[iy_ne_safe * inp_sH + ix_ne_safe * inp_sW] * ne;
            cv += inp_ptr_NC2[iy_sw_safe * inp_sH + ix_sw_safe * inp_sW] * sw;
            cv += inp_ptr_NC2[iy_se_safe * inp_sH + ix_se_safe * inp_sW] * se;

            if (feat_fusion == FeatFusion::MUL)
              for (index_t lki=1; lki<inp_K; ++lki)
                buff_s[((lki+ki)%inp_K)*inp_C + ci] *= cv; // [(lki+ki)%K, tid]
          }
          // __sync_threads();
        
        } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
          index_t ix_nearest = static_cast<index_t>(std::nearbyint(ix));
          index_t iy_nearest = static_cast<index_t>(std::nearbyint(iy));

          const bool ib = within_bounds_2d(iy_nearest, ix_nearest, inp_H, inp_W);

          for (index_t ci=tid; ci<inp_C; ci+=nthreads){
            auto inp_ptr_NC = kpl_param.data + n * inp_sN + ki * inp_sK + ci * inp_sC;

            scalar_t cv = static_cast<scalar_t>(0);
            if (ib)
              cv = inp_ptr_NC[iy_nearest * inp_sH + ix_nearest * inp_sW];

            if (feat_fusion == FeatFusion::MUL)
              for (index_t lki=1; lki<inp_K; ++lki)
                buff_s[((lki+ki)%inp_K)*inp_C + ci] *= cv; // [(lki+ki)%K, tid]
          }
          // __sync_threads();

        }
      }
      }

      // k-plane set grad
      for (index_t ki=0; ki<inp_K; ++ki){
        // get the corresponding input x, y co-ordinates from grid
        scalar_t x = grid.data[grid_offset + ki*grid_sK];
        scalar_t y = grid.data[grid_offset + ki*grid_sK + grid_sCoor];

        // multipliers for gradients on ix and iy
        scalar_t gix_mult, giy_mult;
        scalar_t ix = grid_sampler_compute_source_index_set_grad(x, inp_W, padding_mode, align_corners, &gix_mult);
        scalar_t iy = grid_sampler_compute_source_index_set_grad(y, inp_H, padding_mode, align_corners, &giy_mult);

        if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
          // get NE, NW, SE, SW pixel values from (x, y)
          const index_t ix_nw = static_cast<index_t>(std::floor(ix));
          const index_t iy_nw = static_cast<index_t>(std::floor(iy));
          const index_t ix_ne = ix_nw + 1;
          const index_t iy_ne = iy_nw;
          const index_t ix_sw = ix_nw;
          const index_t iy_sw = iy_nw + 1;
          const index_t ix_se = ix_nw + 1;
          const index_t iy_se = iy_nw + 1;

          // get surfaces to each neighbor:
          scalar_t nw = (ix_se - ix)    * (iy_se - iy);
          scalar_t ne = (ix    - ix_sw) * (iy_sw - iy);
          scalar_t sw = (ix_ne - ix)    * (iy    - iy_ne);
          scalar_t se = (ix    - ix_nw) * (iy    - iy_nw);

          const scalar_t nw_dx = (iy_se - iy);
          const scalar_t nw_dy = (ix_se - ix);
          const scalar_t ne_dx = (iy_sw - iy);
          const scalar_t ne_dy = (ix - ix_sw);
          const scalar_t sw_dx = (iy - iy_ne);
          const scalar_t sw_dy = (ix_ne - ix);
          const scalar_t se_dx = (iy - iy_nw);
          const scalar_t se_dy = (ix - ix_nw);

          const bool ib_nw = within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W);
          const bool ib_ne = within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W);
          const bool ib_sw = within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W);
          const bool ib_se = within_bounds_2d(iy_se, ix_se, inp_H, inp_W);

          const index_t gKpl_off_nw = iy_nw * gKpl_sH + ix_nw * gKpl_sW;
          const index_t gKpl_off_ne = iy_ne * gKpl_sH + ix_ne * gKpl_sW;
          const index_t gKpl_off_sw = iy_sw * gKpl_sH + ix_sw * gKpl_sW;
          const index_t gKpl_off_se = iy_se * gKpl_sH + ix_se * gKpl_sW;
          const index_t inp_off_nw  = iy_nw * inp_sH  + ix_nw * inp_sW;
          const index_t inp_off_ne  = iy_ne * inp_sH  + ix_ne * inp_sW;
          const index_t inp_off_sw  = iy_sw * inp_sH  + ix_sw * inp_sW;
          const index_t inp_off_se  = iy_se * inp_sH  + ix_se * inp_sW;

          auto inp_ptr_NC = kpl_param.data + n * inp_sN + ki * inp_sK;

          float gix = static_cast<float>(0), giy = static_cast<float>(0);

          // const scalar_t *gOut_ptr_NCHW = grad_output.data + n * gOut_sN + h * gOut_sH + w * gOut_sW;
          // index_t NC_offset = n * gInp_sN;
          // for (index_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, NC_offset += gInp_sC, gOut_ptr_NCHW += gOut_sC) {

          index_t inp_NKC_base  = n * inp_sN  + ki * inp_sK  + tid * inp_sC;
          index_t gkpl_NKC_base = n * gKpl_sN + ki * gKpl_sK + tid * gKpl_sC;

          for (index_t ci=tid; ci<inp_C; ci+=nthreads){
            // const scalar_t gOut = *gOut_ptr_NCHW;

            const scalar_t gOut = buff_s[ki*inp_C + ci];

            if (kpl_requires_grad) {
              // calculate and set grad_input. See Note [Passing pointer and offset to fastAtomicAdd].

              if (ib_nw)
                fastAtomicAdd(grad_kpl.data, gkpl_NKC_base + gKpl_off_nw, grad_kpl_memory_span, nw*gOut, true);
              if (ib_ne)
                fastAtomicAdd(grad_kpl.data, gkpl_NKC_base + gKpl_off_ne, grad_kpl_memory_span, ne*gOut, true);
              if (ib_sw)
                fastAtomicAdd(grad_kpl.data, gkpl_NKC_base + gKpl_off_sw, grad_kpl_memory_span, sw*gOut, true);
              if (ib_se)
                fastAtomicAdd(grad_kpl.data, gkpl_NKC_base + gKpl_off_se, grad_kpl_memory_span, se*gOut, true);

              // auto NC_offset              = gkpl_offset;
              // auto grad_input_memory_span = grad_kpl_memory_span;
              // auto gInp_sH = gKpl_sH;
              // auto gInp_sW = gKpl_sW;

              // safe_add_2d(grad_kpl.data, iy_nw, ix_nw, gInp_sH, gInp_sW, inp_H, inp_W, nw * gOut, NC_offset, grad_input_memory_span);
              // safe_add_2d(grad_kpl.data, iy_ne, ix_ne, gInp_sH, gInp_sW, inp_H, inp_W, ne * gOut, NC_offset, grad_input_memory_span);
              // safe_add_2d(grad_kpl.data, iy_sw, ix_sw, gInp_sH, gInp_sW, inp_H, inp_W, sw * gOut, NC_offset, grad_input_memory_span);
              // safe_add_2d(grad_kpl.data, iy_se, ix_se, gInp_sH, gInp_sW, inp_H, inp_W, se * gOut, NC_offset, grad_input_memory_span);
            }

            // calculate grad_grid
            if (grid_requires_grad) {
              if (ib_nw) {
                const scalar_t nw_val = kpl_param.data[inp_NKC_base + inp_off_nw];
                gix -= nw_val * nw_dx * gOut;
                giy -= nw_val * nw_dy * gOut;
              }
              if (ib_ne) {
                const scalar_t ne_val = kpl_param.data[inp_NKC_base + inp_off_ne];
                gix += ne_val * ne_dx * gOut;
                giy -= ne_val * ne_dy * gOut;
              }
              if (ib_sw) {
                const scalar_t sw_val = kpl_param.data[inp_NKC_base + inp_off_sw];
                gix -= sw_val * sw_dx * gOut;
                giy += sw_val * sw_dy * gOut;
              }
              if (ib_se) {
                const scalar_t se_val = kpl_param.data[inp_NKC_base + inp_off_se];
                gix += se_val * se_dx * gOut;
                giy += se_val * se_dy * gOut;
              }

            }
            // if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
            //   scalar_t nw_val = inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW];
            //   gix -= nw_val * (iy_se - iy) * gOut;
            //   giy -= nw_val * (ix_se - ix) * gOut;
            // }
            // if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
            //   scalar_t ne_val = inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW];
            //   gix += ne_val * (iy_sw - iy) * gOut;
            //   giy -= ne_val * (ix - ix_sw) * gOut;
            // }
            // if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
            //   scalar_t sw_val = inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW];
            //   gix -= sw_val * (iy - iy_ne) * gOut;
            //   giy += sw_val * (ix_ne - ix) * gOut;
            // }
            // if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
            //   scalar_t se_val = inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW];
            //   gix += se_val * (iy - iy_nw) * gOut;
            //   giy += se_val * (ix - ix_nw) * gOut;
            // }

            inp_NKC_base  += nthreads * inp_sC;
            gkpl_NKC_base += nthreads * gKpl_sC;
          }

          // assuming grad_grid is contiguous
          // thus we can
          //   1. use index with gGrid_sW to directly compute gGrid_ptr_NHW
          //   2. directly assign to gGrid_ptr_NHW[0], gGrid_ptr_NHW[1]
          // scalar_t *gGrid_ptr_NHW = grad_grid.data + index * gGrid_sW;
          // gGrid_ptr_NHW[0] = gix_mult * gix;
          // gGrid_ptr_NHW[1] = giy_mult * giy;
          
          if (grid_requires_grad){
            // TODO: parallel reduce

            float gx = static_cast<float>(gix);
            float gy = static_cast<float>(giy);

            gx += __shfl_down_sync(static_cast<unsigned int>(0xffffffff), gx, static_cast<int>(16));
            gx += __shfl_down_sync(static_cast<unsigned int>(0xffffffff), gx, static_cast<int>( 8));
            gx += __shfl_down_sync(static_cast<unsigned int>(0xffffffff), gx, static_cast<int>( 4));
            gx += __shfl_down_sync(static_cast<unsigned int>(0xffffffff), gx, static_cast<int>( 2));
            gx += __shfl_down_sync(static_cast<unsigned int>(0xffffffff), gx, static_cast<int>( 1));

            gy += __shfl_down_sync(static_cast<unsigned int>(0xffffffff), gy, static_cast<int>(16));
            gy += __shfl_down_sync(static_cast<unsigned int>(0xffffffff), gy, static_cast<int>( 8));
            gy += __shfl_down_sync(static_cast<unsigned int>(0xffffffff), gy, static_cast<int>( 4));
            gy += __shfl_down_sync(static_cast<unsigned int>(0xffffffff), gy, static_cast<int>( 2));
            gy += __shfl_down_sync(static_cast<unsigned int>(0xffffffff), gy, static_cast<int>( 1));

            if (tid == 0){
              grad_grid.data[gGrid_offset + ki*gGrid_sK]               = gix_mult * gx;
              grad_grid.data[gGrid_offset + ki*gGrid_sK + gGrid_sCoor] = giy_mult * gy;
            }
          }

        } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {

          index_t ix_nearest = static_cast<index_t>(std::nearbyint(ix));
          index_t iy_nearest = static_cast<index_t>(std::nearbyint(iy));

          const bool ib = within_bounds_2d(iy_nearest, ix_nearest, inp_H, inp_W);

          if (kpl_requires_grad && ib) {
            index_t gkpl_offset = n * gKpl_sN + ki * gKpl_sK + \
                                  iy_nearest * gKpl_sH + ix_nearest * gKpl_sW;

            // assign nearest neighbour pixel value to output pixel
            for (index_t ci=tid; ci<inp_C; ci+=nthreads){
              fastAtomicAdd(grad_kpl.data, gkpl_offset, grad_kpl_memory_span, buff_s[ki*inp_C + ci], true);
              gkpl_offset += nthreads * gKpl_sC;
            }
          }

          // assuming grad_grid is contiguous
          // thus we can
          //   1. use index with gGrid_sW to directly compute gGrid_ptr_NHW
          //   2. directly assign to gGrid_ptr_NHW[0], gGrid_ptr_NHW[1]
          // scalar_t *gGrid_ptr_NHW = grad_grid.data + index * gGrid_sW;
          // gGrid_ptr_NHW[0] = static_cast<scalar_t>(0);
          // gGrid_ptr_NHW[1] = static_cast<scalar_t>(0);
        }
      }
    }
  }


} // namespace

torch::Tensor kplane_mlp_forward_cuda(
// void kplane_mlp_forward_cuda(torch::Tensor& output,
    const torch::Tensor &kpl_param, const torch::Tensor &mlp_param,  
    const torch::Tensor &grid,      const torch::Tensor &skip,      

    int64_t interpolation_mode, int64_t padding_mode, bool align_corners, int64_t feature_fusion,
    
    int64_t mlp_layers, int64_t mlp_dim_hidden, int64_t mlp_dim_output, int64_t mlp_activation
) {
  // int64_t mlp_layers = 2, mlp_dim_hidden = 64, mlp_dim_output = 32, mlp_activation = 0;

  // auto kpl_input = at::permute(kpl_param, {0, 3, 4, 1, 2}).contiguous();
  auto kpl_input = kpl_param;

  auto N = kpl_input.size(0);
  auto K = kpl_input.size(3);
  auto C = kpl_input.size(4);

  auto skip_C = skip.size(3);

  int64_t H = grid.size(1); // output H
  int64_t W = grid.size(2); // output W
  int64_t count = N * H * W;

  const int64_t kpl_C = C + skip_C;
  const int64_t out_C = mlp_layers == 0 ? kpl_C : mlp_dim_output;

  const int64_t max_C = std::max(kpl_C, std::max(mlp_dim_hidden, out_C));

  auto output = torch::empty({N, H, W, out_C}, kpl_param.options());

  dim3 blocks = {(H*W-1)/WARP_CHUNK_K + 1, N};

  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES(
      kpl_input.scalar_type(), "kplane_mlp_forward_cuda", [&] {
      if (canUse32BitIndexMath(kpl_input) && canUse32BitIndexMath(grid) &&
          canUse32BitIndexMath(output)) {

        int num_kpl  = sizeof(float) * (WARP_CHUNK_K*kpl_C);
        int num_out  = sizeof(float) * (WARP_CHUNK_K*max_C);

      int num_wmma = 2 * (WARP_CHUNK_K*64 + 64*64);
      #if SHMEM_WMMA_BUDGET == 0
      #else
        num_wmma = mlp_layers>0 ? num_wmma : 0;
      #endif
        
        int align = 16;
        int num_kpl_skew = num_kpl % align ? 0 : (align - num_kpl%align) ; // 8-byte align
        int num_out_skew = num_out % align ? 0 : (align - num_out%align) ; // 8-byte align

        int shmem_size = num_kpl + num_out + num_kpl_skew + num_out_skew + num_wmma;

        // cudaFuncSetAttribute(kplane_mlp_forward_kernel<scalar_t, int>, cudaFuncAttributeMaxDynamicSharedMemorySize , shmem_size);
        cudaFuncSetCacheConfig(kplane_mlp_forward_kernel<scalar_t, int>, cudaFuncCachePreferShared);
        kplane_mlp_forward_kernel<scalar_t>
          <<<blocks, 32, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
            static_cast<int>(32),
            num_kpl_skew,
            num_out_skew,

            #if USE_CONST_TI > 0
            getTensorInfo<const scalar_t, int>(kpl_input),
            getTensorInfo<const scalar_t, int>(mlp_param),
            getTensorInfo<const scalar_t, int>(grid),
            getTensorInfo<const scalar_t, int>(skip),
            #else
            getTensorInfo<scalar_t, int>(kpl_input),
            getTensorInfo<scalar_t, int>(mlp_param),
            getTensorInfo<scalar_t, int>(grid),
            getTensorInfo<scalar_t, int>(skip),
            #endif
            getTensorInfo<scalar_t, int>(output),

            static_cast<GridSamplerInterpolation>(interpolation_mode),
            static_cast<GridSamplerPadding>(padding_mode),
            align_corners, 
            static_cast<FeatFusion>(feature_fusion),

            static_cast<int>(mlp_layers), static_cast<int>(mlp_dim_hidden), static_cast<int>(mlp_dim_output),
            static_cast<Activation>(mlp_activation)
            );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {

        int num_kpl  = sizeof(float) * (WARP_CHUNK_K*kpl_C);
        int num_out  = sizeof(float) * (WARP_CHUNK_K*max_C);

      int num_wmma = 2 * (WARP_CHUNK_K*64 + 64*64);
      #if SHMEM_WMMA_BUDGET == 0
      #else
        num_wmma = mlp_layers>0 ? num_wmma : 0;
      #endif
        
        int64_t align = 16;
        int64_t num_kpl_skew = num_kpl % align ? 0 : (align - num_kpl%align) ; // 8-byte align
        int64_t num_out_skew = num_out % align ? 0 : (align - num_out%align) ; // 8-byte align

        int shmem_size = num_kpl + num_out + num_kpl_skew + num_out_skew + num_wmma;

        // cudaFuncSetAttribute(kplane_mlp_forward_kernel<scalar_t, int>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
        cudaFuncSetCacheConfig(kplane_mlp_forward_kernel<scalar_t, int64_t>, cudaFuncCachePreferShared);
        kplane_mlp_forward_kernel<scalar_t>
          <<<blocks, 32, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
            static_cast<int64_t>(32), 
            num_kpl_skew,
            num_out_skew,

            #if USE_CONST_TI > 0
            getTensorInfo<const scalar_t, int64_t>(kpl_input),
            getTensorInfo<const scalar_t, int64_t>(mlp_param),
            getTensorInfo<const scalar_t, int64_t>(grid),
            getTensorInfo<const scalar_t, int64_t>(skip),
            #else
            getTensorInfo<scalar_t, int64_t>(kpl_input),
            getTensorInfo<scalar_t, int64_t>(mlp_param),
            getTensorInfo<scalar_t, int64_t>(grid),
            getTensorInfo<scalar_t, int64_t>(skip),
            #endif
            getTensorInfo<scalar_t, int64_t>(output),

            static_cast<GridSamplerInterpolation>(interpolation_mode),
            static_cast<GridSamplerPadding>(padding_mode),
            align_corners, 
            static_cast<FeatFusion>(feature_fusion),
            
            mlp_layers, mlp_dim_hidden, mlp_dim_output,
            static_cast<Activation>(mlp_activation)
            );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    });
  }

  return output;
}

std::vector<torch::Tensor> kplane_mlp_backward_cuda(
// void kplane_mlp_backward_cuda(torch::Tensor &grad_kpl, torch::Tensor &grad_mlp, torch::Tensor &grad_grid, torch::Tensor &grad_skip,
    const torch::Tensor &grad_output,

    const torch::Tensor &kpl_param, const torch::Tensor &mlp_param,  
    const torch::Tensor &grid,      const torch::Tensor &skip,      

    int64_t interpolation_mode, int64_t padding_mode, bool align_corners, int64_t feature_fusion,
    
    int64_t mlp_layers, int64_t mlp_dim_hidden, int64_t mlp_dim_output, int64_t mlp_activation,

    std::array<bool, 4> output_mask
) {

  // auto kpl_input = at::permute(kpl_param, {0, 3, 4, 1, 2}).contiguous();
  // auto kpl_input = kpl_param;
  const torch::Tensor &kpl_input = kpl_param;

  auto N = kpl_input.size(0);
  auto K = kpl_input.size(3);
  auto C = kpl_input.size(4);

  auto skip_C = skip.size(3);

  auto H = grid.size(1); // output H
  auto W = grid.size(2); // output W
  int64_t count = N * H * W;

  bool kpl_requires_grad  = output_mask[0];
  bool mlp_requires_grad  = output_mask[1];
  bool grid_requires_grad = output_mask[2];
  bool skip_requires_grad = output_mask[3];

  // torch::Tensor grad_kpl  = torch::zeros_like(kpl_param);
  // torch::Tensor grad_mlp  = torch::zeros_like(mlp_param);
  // torch::Tensor grad_grid = torch::zeros_like(grid);
  // torch::Tensor grad_skip = torch::zeros_like(skip);

  auto grad_kpl = ([&](){
    if (kpl_requires_grad)
      return torch::zeros_like(kpl_param);
    else
      return torch::Tensor();
  })();

  auto grad_mlp = ([&](){
    if (mlp_requires_grad)
      return torch::empty_like(mlp_param);
    else
      return torch::Tensor();
  })();

  auto grad_grid = ([&](){
    if (grid_requires_grad)
      if (static_cast<GridSamplerInterpolation>(interpolation_mode) == GridSamplerInterpolation::Nearest)
        return torch::zeros_like(grid);
      else
        return torch::empty_like(grid);
    else
      return torch::Tensor();
  })();

  auto grad_skip = ([&](){
    if (skip_requires_grad)
      return torch::empty_like(skip);
    else
      return torch::Tensor();
  })();

  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES(
      kpl_input.scalar_type(), "kplane_mlp_backward_cuda", [&] {
      if (canUse32BitIndexMath(kpl_input) && canUse32BitIndexMath(grid) &&
          canUse32BitIndexMath(grad_output)) {
        
        int shmem_size = sizeof(scalar_t) * (K*C+skip_C);

        // cudaFuncSetAttribute(kplane_mlp_backward_kernel<scalar_t, int>, cudaFuncAttributeMaxDynamicSharedMemorySize , shmem_size);
        // cudaFuncSetCacheConfig(kplane_mlp_backward_kernel<scalar_t, int>, cudaFuncCachePreferShared);
        kplane_mlp_backward_kernel<scalar_t>
          <<<GET_BLOCKS(count, 1), 32, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
            static_cast<int>(32),

            #if USE_CONST_TI > 0
            getTensorInfo<const scalar_t, int>(grad_output),
            getTensorInfo<const scalar_t, int>(kpl_input),
            getTensorInfo<const scalar_t, int>(mlp_param),
            getTensorInfo<const scalar_t, int>(grid),
            getTensorInfo<const scalar_t, int>(skip),
            #else
            getTensorInfo<scalar_t, int>(grad_output),
            getTensorInfo<scalar_t, int>(kpl_input),
            getTensorInfo<scalar_t, int>(mlp_param),
            getTensorInfo<scalar_t, int>(grid),
            getTensorInfo<scalar_t, int>(skip),
            #endif

            kpl_requires_grad  ? getTensorInfo<scalar_t, int>(grad_kpl)  : TensorInfo<scalar_t, int>(),
            mlp_requires_grad  ? getTensorInfo<scalar_t, int>(grad_mlp)  : TensorInfo<scalar_t, int>(),
            grid_requires_grad ? getTensorInfo<scalar_t, int>(grad_grid) : TensorInfo<scalar_t, int>(),
            skip_requires_grad ? getTensorInfo<scalar_t, int>(grad_skip) : TensorInfo<scalar_t, int>(),
            // getTensorInfo<scalar_t, int>(grad_kpl)  ,
            // getTensorInfo<scalar_t, int>(grad_mlp)  ,
            // getTensorInfo<scalar_t, int>(grad_grid) ,
            // getTensorInfo<scalar_t, int>(grad_skip) ,

            static_cast<GridSamplerInterpolation>(interpolation_mode),
            static_cast<GridSamplerPadding>(padding_mode),
            align_corners, 
            static_cast<FeatFusion>(feature_fusion),

            static_cast<int>(mlp_layers), static_cast<int>(mlp_dim_hidden), static_cast<int>(mlp_dim_output),
            static_cast<Activation>(mlp_activation),

            kpl_requires_grad  ? static_cast<int>(grad_kpl.numel() ) : 0,
            mlp_requires_grad  ? static_cast<int>(grad_mlp.numel() ) : 0,
            grid_requires_grad ? static_cast<int>(grad_grid.numel()) : 0,
            skip_requires_grad ? static_cast<int>(grad_skip.numel()) : 0
            );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {

        int shmem_size = sizeof(scalar_t) * (K*C+skip_C);

        // cudaFuncSetAttribute(kplane_mlp_backward_kernel<scalar_t, int>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
        // cudaFuncSetCacheConfig(kplane_mlp_backward_kernel<scalar_t, int64_t>, cudaFuncCachePreferShared);
        kplane_mlp_backward_kernel<scalar_t>
          <<<GET_BLOCKS(count, 1), 32, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
            static_cast<int64_t>(32), 

            #if USE_CONST_TI > 0
            getTensorInfo<const scalar_t, int64_t>(grad_output),
            getTensorInfo<const scalar_t, int64_t>(kpl_input),
            getTensorInfo<const scalar_t, int64_t>(mlp_param),
            getTensorInfo<const scalar_t, int64_t>(grid),
            getTensorInfo<const scalar_t, int64_t>(skip),
            #else
            getTensorInfo<scalar_t, int64_t>(grad_output),
            getTensorInfo<scalar_t, int64_t>(kpl_input),
            getTensorInfo<scalar_t, int64_t>(mlp_param),
            getTensorInfo<scalar_t, int64_t>(grid),
            getTensorInfo<scalar_t, int64_t>(skip),
            #endif

            kpl_requires_grad  ? getTensorInfo<scalar_t, int64_t>(grad_kpl)  : TensorInfo<scalar_t, int64_t>(),
            mlp_requires_grad  ? getTensorInfo<scalar_t, int64_t>(grad_mlp)  : TensorInfo<scalar_t, int64_t>(),
            grid_requires_grad ? getTensorInfo<scalar_t, int64_t>(grad_grid) : TensorInfo<scalar_t, int64_t>(),
            skip_requires_grad ? getTensorInfo<scalar_t, int64_t>(grad_skip) : TensorInfo<scalar_t, int64_t>(),
            // getTensorInfo<scalar_t, int64_t>(grad_kpl)  ,
            // getTensorInfo<scalar_t, int64_t>(grad_mlp)  ,
            // getTensorInfo<scalar_t, int64_t>(grad_grid) ,
            // getTensorInfo<scalar_t, int64_t>(grad_skip) ,

            static_cast<GridSamplerInterpolation>(interpolation_mode),
            static_cast<GridSamplerPadding>(padding_mode),
            align_corners, 
            static_cast<FeatFusion>(feature_fusion),
            
            mlp_layers, mlp_dim_hidden, mlp_dim_output,
            static_cast<Activation>(mlp_activation),

            kpl_requires_grad  ? grad_kpl.numel()  : 0,
            mlp_requires_grad  ? grad_mlp.numel()  : 0,
            grid_requires_grad ? grad_grid.numel() : 0,
            skip_requires_grad ? grad_skip.numel() : 0
            );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    });
  }

  // auto kpl_input = at::permute(kpl_param, {0, 3, 4, 1, 2}).contiguous();

  return {grad_kpl, grad_mlp, grad_grid, grad_skip};
}

}  // namespace at::native