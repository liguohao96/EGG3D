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

#include "common.h"
#include "cuda_activation.cuh"
#include "cuda_fc.cuh"

using namespace nvcuda;

namespace at::native {

using namespace at::cuda::detail;

using at::native::detail::GridSamplerInterpolation;
using at::native::detail::GridSamplerPadding;

// MODIFIED BASED ON: https://github.com/pytorch/pytorch/blob/31bb65de195d5c3cc5d74e7b909495ef8c2035e7/aten/src/ATen/native/cuda/GridSampler.cu

#define INFO_SHMEM 0 // print shmem info before kernel launch

#define WARP_SIZE 32

#define ALIGN_BYTES 32

#define MLP_CHUNK_SIZE 8 // chunk size of mlp

#if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 3
#define USE_CONST_TI 1
#else
#define USE_CONST_TI 0
#endif

#define USE_TEMPLATE_ARGS 1

#define DISPATCH_ENUM_CASE(NAME, VALUE, ...) \
    case (VALUE) : {                               \
      constexpr auto NAME = VALUE;                 \
      __VA_ARGS__();                               \
      break;                                       \
    }                                              \

#define DISPATCH_INTERPOLATION(VALUE, ...)                                               \
  switch(VALUE){                                                                         \
    DISPATCH_ENUM_CASE(Interpolation_E, GridSamplerInterpolation::Bilinear, __VA_ARGS__) \
    DISPATCH_ENUM_CASE(Interpolation_E, GridSamplerInterpolation::Nearest,  __VA_ARGS__) \
  }                                                                                      \

#define DISPATCH_PADDING(VALUE, ...)                                           \
  switch(VALUE){                                                               \
    DISPATCH_ENUM_CASE(Padding_E, GridSamplerPadding::Zeros,      __VA_ARGS__) \
    DISPATCH_ENUM_CASE(Padding_E, GridSamplerPadding::Border,     __VA_ARGS__) \
    DISPATCH_ENUM_CASE(Padding_E, GridSamplerPadding::Reflection, __VA_ARGS__) \
  }                                                                            \

#define DISPATCH_ALIGN(VALUE, ...)                  \
  switch(VALUE){                                    \
    DISPATCH_ENUM_CASE(Align_E, true,  __VA_ARGS__) \
    DISPATCH_ENUM_CASE(Align_E, false, __VA_ARGS__) \
  }                                                 \

#define DISPATCH_FUSHION(VALUE, ...)                            \
  switch(VALUE){                                                \
    DISPATCH_ENUM_CASE(Fushion_E, FeatFusion::SUM, __VA_ARGS__) \
    DISPATCH_ENUM_CASE(Fushion_E, FeatFusion::AVG, __VA_ARGS__) \
    DISPATCH_ENUM_CASE(Fushion_E, FeatFusion::MUL, __VA_ARGS__) \
  }                                                             \

#define DISPATCH_ACTIVATION(VALUE, ...)                            \
  switch(VALUE){                                                        \
    DISPATCH_ENUM_CASE(Activation_E, Activation::Softplus, __VA_ARGS__) \
    DISPATCH_ENUM_CASE(Activation_E, Activation::ReLU,     __VA_ARGS__) \
    DISPATCH_ENUM_CASE(Activation_E, Activation::None,     __VA_ARGS__) \
  }                                                                     \

namespace {

  template <GridSamplerInterpolation interpolation_mode, GridSamplerPadding padding_mode, bool align_corners, FeatFusion feat_fusion>
  __global__ void TemplateKernel(){
    printf("template = %d %d %d %d\n", interpolation_mode, padding_mode, align_corners, feat_fusion);
  };

  #if USE_TEMPLATE_ARGS
  template <typename scalar_t, typename index_t,
    GridSamplerInterpolation interpolation_mode, GridSamplerPadding padding_mode, bool align_corners, FeatFusion feat_fusion>
  #else
  template <typename scalar_t, typename index_t>
  #endif
  __launch_bounds__(MLP_CHUNK_SIZE*WARP_SIZE/4) // MLP_CHUNK_SIZE/4 * 32
  __global__ void kplane_forward_kernel(

    #if USE_CONST_TI > 0
      TensorInfo<const scalar_t, index_t> kpl_param, // B, H, W, K, C
      TensorInfo<const scalar_t, index_t> grid,      // B, p, q, K, 2
      TensorInfo<const scalar_t, index_t> skip,      // B, p, q, sC
    #else
      TensorInfo<scalar_t, index_t> kpl_param, // B, H, W, K, C
      TensorInfo<scalar_t, index_t> grid,      // B, p, q, K, 2
      TensorInfo<scalar_t, index_t> skip,      // B, p, q, sC
    #endif

    #if USE_TEMPLATE_ARGS
      TensorInfo<scalar_t, index_t> output           // B, p, q, oC
    #else

      TensorInfo<scalar_t, index_t> output,          // B, p, q, oC

      const GridSamplerInterpolation interpolation_mode,
      const GridSamplerPadding padding_mode,
      const bool align_corners,
      const FeatFusion feat_fusion
    #endif
      
      ) {

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
    // scalar shmem_kpl_ptr[MLP_CHUNK_SIZE][inp_C + skip_C]

    const index_t tid  = threadIdx.x;  // thread id
    const index_t cid  = threadIdx.y;  // chunk  id
    const index_t n    = blockIdx.y;   // batch  id

    const index_t chunk_stride = blockDim.y;

    const index_t kpl_C = inp_C + skip_C;

    const int32_t kpl_base  = 0;

    float* shmem_kpl_ptr = (float*)(shmem);            // scalar shmem_kpl_ptr[MLP_CHUNK_SIZE][inp_C + skip_C] # 8*(<64) = <0.5K

    const scalar_t* __restrict__ kpl_data  = kpl_param.data;
    const scalar_t* __restrict__ skip_data = skip.data;
    const scalar_t* __restrict__ grid_data = grid.data;

    // init
    {
      float init_value = static_cast<float>(0);
      #if USE_TEMPLATE_ARGS
      if constexpr (feat_fusion == FeatFusion::MUL)
        init_value = static_cast<float>(1);
      #else
      if (feat_fusion == FeatFusion::MUL)
        init_value = static_cast<float>(1);
      #endif

      #pragma unroll
      for (int32_t item_i=cid; item_i<MLP_CHUNK_SIZE; item_i+=chunk_stride)  // not looped at all
        for (int32_t ci=tid; ci<kpl_C; ci+=WARP_SIZE)
          shmem_kpl_ptr[item_i*(kpl_C) + ci] = init_value;
    }

    // for (int32_t item_i=0; item_i<MLP_CHUNK_SIZE; ++item_i){

    // #pragma unroll
    for (int32_t item_i=cid; item_i<MLP_CHUNK_SIZE; item_i+=chunk_stride){ // not looped at all
      const index_t index = blockIdx.x*MLP_CHUNK_SIZE + item_i;

      if (index > out_W*out_H)
        break;

      float* buff_s = shmem_kpl_ptr + item_i*kpl_C;

      const index_t w = index % out_W;
      const index_t h = (index / out_W) % out_H;
      const index_t grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;

      // concatenate
      {
        auto skip_offset = n*skip_sN + h*skip_sH + w*skip_sW;
        for (index_t i=tid; i<skip_C; i+=WARP_SIZE)
          buff_s[inp_C+i] = skip_data[skip_offset + i*skip_sC];
      }
      
      // __sync_threads();

      const scalar_t* __restrict__ grid_ptr_NHW = grid_data + grid_offset;
      // k-plane feature query
      for (index_t ki=0; ki<inp_K; ++ki){
        // get the corresponding input x, y co-ordinates from grid
        // scalar_t x = grid_data[grid_offset + ki*grid_sK];
        // scalar_t y = grid_data[grid_offset + ki*grid_sK + grid_sCoor];
        scalar_t x = grid_ptr_NHW[ki*grid_sK];
        scalar_t y = grid_ptr_NHW[ki*grid_sK + grid_sCoor];

        scalar_t ix = grid_sampler_compute_source_index(x, inp_W, padding_mode, align_corners);
        scalar_t iy = grid_sampler_compute_source_index(y, inp_H, padding_mode, align_corners);

        #if USE_TEMPLATE_ARGS
        if constexpr (interpolation_mode == GridSamplerInterpolation::Bilinear)
        #else
        if (interpolation_mode == GridSamplerInterpolation::Bilinear)
        #endif
        {
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

          const scalar_t* __restrict__ inp_ptr_NC = kpl_param.data + n * inp_sN + ki * inp_sK;

          for (index_t ci=tid; ci<inp_C; ci+=WARP_SIZE){
            // calculate bilinear weighted pixel value and set output pixel

            const scalar_t* __restrict__ inp_ptr_NC2 = inp_ptr_NC + ci * inp_sC;

            scalar_t cv = static_cast<scalar_t>(0);
            cv += inp_ptr_NC2[iy_nw_safe * inp_sH + ix_nw_safe * inp_sW] * nw;
            cv += inp_ptr_NC2[iy_ne_safe * inp_sH + ix_ne_safe * inp_sW] * ne;
            cv += inp_ptr_NC2[iy_sw_safe * inp_sH + ix_sw_safe * inp_sW] * sw;
            cv += inp_ptr_NC2[iy_se_safe * inp_sH + ix_se_safe * inp_sW] * se;

            #if USE_TEMPLATE_ARGS
            if constexpr (feat_fusion == FeatFusion::SUM)
              buff_s[ci] += cv;
            else if constexpr (feat_fusion == FeatFusion::AVG)
              buff_s[ci] += cv/inp_K;
            else if constexpr (feat_fusion == FeatFusion::MUL)
              buff_s[ci] *= cv;
            #else
            if (feat_fusion == FeatFusion::SUM)
              buff_s[ci] += cv;
            else if (feat_fusion == FeatFusion::AVG)
              buff_s[ci] += cv/inp_K;
            else if (feat_fusion == FeatFusion::MUL)
              buff_s[ci] *= cv;
            #endif
          }
          // __sync_threads();
        } 
        #if USE_TEMPLATE_ARGS
        else if constexpr (interpolation_mode == GridSamplerInterpolation::Nearest) 
        #else
        else if (interpolation_mode == GridSamplerInterpolation::Nearest)
        #endif
        {
          index_t ix_nearest = static_cast<index_t>(std::nearbyint(ix));
          index_t iy_nearest = static_cast<index_t>(std::nearbyint(iy));

          const bool ib = within_bounds_2d(iy_nearest, ix_nearest, inp_H, inp_W);

          for (index_t ci=tid; ci<inp_C; ci+=WARP_SIZE){
            const scalar_t* __restrict__ inp_ptr_NC = kpl_param.data + n * inp_sN + ki * inp_sK + ci * inp_sC;

            scalar_t cv = static_cast<scalar_t>(0);
            if (ib)
              cv = inp_ptr_NC[iy_nearest * inp_sH + ix_nearest * inp_sW];

          #if USE_TEMPLATE_ARGS
            if constexpr (feat_fusion == FeatFusion::SUM)
              buff_s[ci] += cv;
            else if constexpr (feat_fusion == FeatFusion::AVG)
              buff_s[ci] += cv/inp_K;
            else if constexpr (feat_fusion == FeatFusion::MUL)
              buff_s[ci] *= cv;
          #else
            if (feat_fusion == FeatFusion::SUM)
              buff_s[ci] += cv;
            else if (feat_fusion == FeatFusion::AVG)
              buff_s[ci] += cv/inp_K;
            else if (feat_fusion == FeatFusion::MUL)
              buff_s[ci] *= cv;
          #endif
          }
          // __syncthreads();

        }
      }

    }

    for (int32_t item_i=cid; item_i<MLP_CHUNK_SIZE; item_i += chunk_stride){
      const index_t index = blockIdx.x*MLP_CHUNK_SIZE + item_i;

      if (index > out_W*out_H)
        return;

      const float* buff_o = shmem_kpl_ptr + item_i*kpl_C;

      const index_t w = index % out_W;
      const index_t h = (index / out_W) % out_H;

      auto output_offset = n*out_sN + h*out_sH + w*out_sW;
      for (index_t i=tid; i<out_C; i+=WARP_SIZE){
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
  __launch_bounds__(WARP_SIZE) // MLP_CHUNK_SIZE/4 * 32
  __global__ void kplane_backward_kernel(

      #if USE_CONST_TI > 0
      TensorInfo<const scalar_t, index_t> grad_output,

      TensorInfo<const scalar_t, index_t> kpl_param, // B, H, W, K, C
      TensorInfo<const scalar_t, index_t> grid,      // B, p, q, K, 2
      TensorInfo<const scalar_t, index_t> skip,      // B, p, q, sC
      #else
      TensorInfo<scalar_t, index_t> grad_output,

      TensorInfo<scalar_t, index_t> kpl_param, // B, H, W, K, C
      TensorInfo<scalar_t, index_t> grid,      // B, p, q, K, 2
      TensorInfo<scalar_t, index_t> skip,      // B, p, q, sC
      #endif

      TensorInfo<scalar_t, index_t> grad_kpl,    // B, H, W, K, C  initialized to zeros (or unused if input_requires_grad is false)
      TensorInfo<scalar_t, index_t> grad_grid,   // B, p, q, K, 2 initialized to empty (0 if interpolation == "nearest")
      TensorInfo<scalar_t, index_t> grad_skip,   // B, p, q, sC   initialized to empty

      const GridSamplerInterpolation interpolation_mode,
      const GridSamplerPadding padding_mode,
      const bool align_corners, 
      const FeatFusion feat_fusion,
      
      const index_t grad_kpl_memory_span,
      const index_t grad_grid_memory_span,
      const index_t grad_skip_memory_span
      ) {
      
    const bool kpl_requires_grad  = grad_kpl_memory_span  > 0;
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

    const index_t tid  = threadIdx.x;  // thread id
    const index_t cid  = threadIdx.y;  // chunk  id
    const index_t n    = blockIdx.y;   // batch  id

    const index_t chunk_stride = blockDim.y;


    extern __shared__ char shmem[];
    // scalar shmem_kpl_ptr[K][MLP_CHUNK_SIZE][inp_C]

    float* shmem_kpl_ptr = (float*)(shmem);            // scalar shmem_kpl_ptr[MLP_CHUNK_SIZE][inp_C + skip_C] # 8*(<64) = <0.5K

    const scalar_t* __restrict__ kpl_data  = kpl_param.data;
    const scalar_t* __restrict__ skip_data = skip.data;
    const scalar_t* __restrict__ grid_data = grid.data;
    const scalar_t* __restrict__ gout_data = grad_output.data;

    // // init
    // {
    //   float init_value = static_cast<float>(0);
    //   if (feat_fusion == FeatFusion::MUL)
    //     init_value = static_cast<float>(1);

    //   cid*inp_K*inp_C + ki*inp_C

    //   for (int32_t item_i=cid; item_i<MLP_CHUNK_SIZE; item_i+=chunk_stride)  // not looped at all
    //     // for (int32_t ci=tid; ci<kpl_C; ci+=WARP_SIZE)
    //     //   shmem_kpl_ptr[item_i*(kpl_C) + ci] = init_value;
    //     scalar_t gOut_val = grad_output.data[gOut_offset + (ci)*gOut_sC];
    //     for (index_t ki=0; ki<inp_K; ++ki)
    //       shmem_kpl_ptr[ki*inp_C+ci] = gOut_val;
      

    // }

    // for(int32_t item_i=cid; item_i<MLP_CHUNK_SIZE; item_i+=chunk_stride){ // not looped at all
    //   const index_t index = blockIdx.x*MLP_CHUNK_SIZE + item_i;

    {
      const index_t index = blockIdx.x;

      const index_t w = index % out_W;
      const index_t h = (index / out_W) % out_H;
      const auto grid_offset  = n * grid_sN  + h * grid_sH  + w * grid_sW;
      const auto gOut_offset  = n * gOut_sN  + h * gOut_sH  + w * gOut_sW;
      const auto gGrid_offset = n * gGrid_sN + h * gGrid_sH + w * gGrid_sW;
      const auto gSkip_offset = n * gSkip_sN + h * gSkip_sH + w * gSkip_sW;

      float* buff_s = (float*)shmem;

      // init buff_s[K][inp_C] = grad[bi,hi,wi,:].expand(K,-1)
      {
        for (index_t ci=tid; ci<inp_C; ci+=WARP_SIZE){
          scalar_t gOut_val = grad_output.data[gOut_offset + (ci)*gOut_sC];
          for (index_t ki=0; ki<inp_K; ++ki)
            buff_s[ki*inp_C+ci] = gOut_val;
        }
      }
      // concatenate
      if (skip_requires_grad) {
        for (index_t i=tid; i<skip_C; i+=WARP_SIZE)
          grad_skip.data[gSkip_offset + i*gSkip_sC] = grad_output.data[gOut_offset + (inp_C+i)*gOut_sC];
      }
      
      // __sync_threads();

      // set input correlated gradient
      if (feat_fusion == FeatFusion::SUM) {
      }
      else if (feat_fusion == FeatFusion::AVG){
        for (index_t ci=tid; ci<inp_K*inp_C; ci+=WARP_SIZE)
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

          for (index_t ci=tid; ci<inp_C; ci+=WARP_SIZE){
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

          for (index_t ci=tid; ci<inp_C; ci+=WARP_SIZE){
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

          for (index_t ci=tid; ci<inp_C; ci+=WARP_SIZE){
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
            inp_NKC_base  += WARP_SIZE * inp_sC;
            gkpl_NKC_base += WARP_SIZE * gKpl_sC;
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
            for (index_t ci=tid; ci<inp_C; ci+=WARP_SIZE){
              fastAtomicAdd(grad_kpl.data, gkpl_offset, grad_kpl_memory_span, (scalar_t)buff_s[ki*inp_C + ci], true);
              gkpl_offset += WARP_SIZE * gKpl_sC;
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

// template<unsigned mlp_activation> 
torch::Tensor kplane_forward_cuda(
// void kplane_mlp_forward_cuda(torch::Tensor& output,
    const torch::Tensor &kpl_param, 
    const torch::Tensor &grid,      const torch::Tensor &skip,      

    const int64_t interpolation_mode, const int64_t padding_mode, const bool align_corners, const int64_t feature_fusion
) {
  auto kpl_input = kpl_param;

  auto N = kpl_input.size(0);
  auto K = kpl_input.size(3);
  auto C = kpl_input.size(4);

  auto skip_C = skip.size(3);

  int64_t H = grid.size(1); // output H
  int64_t W = grid.size(2); // output W
  int64_t count = N * H * W;

  const int64_t kpl_C = C + skip_C;
  const int64_t out_C = kpl_C;

  const int64_t max_C = kpl_C;

  auto output = torch::empty({N, H, W, out_C}, kpl_param.options());

  dim3 blocks  = {(H*W-1)/MLP_CHUNK_SIZE + 1, N};
  dim3 threads = {32, MLP_CHUNK_SIZE/4};

  constexpr unsigned align_float = ALIGN_BYTES / sizeof(float);

  if (count > 0) {

    int temp_C   = max_C%align_float ? max_C : (max_C + align_float - max_C%align_float);
    int num_kpl  = sizeof(float) * (MLP_CHUNK_SIZE*temp_C);

    int num_kpl_skew = num_kpl % ALIGN_BYTES ? 0 : (ALIGN_BYTES - num_kpl%ALIGN_BYTES) ; // 8-byte align

    int shmem_size = num_kpl;

    // FIXME: this is very slow !!! and got wrong result !!!
    DISPATCH_INTERPOLATION(static_cast<GridSamplerInterpolation>(interpolation_mode), ([&]{
    DISPATCH_PADDING(static_cast<GridSamplerPadding>(padding_mode), ([&]{
    DISPATCH_ALIGN(align_corners,                             ([&]{
    DISPATCH_FUSHION(static_cast<FeatFusion>(feature_fusion), ([&]{

    AT_DISPATCH_FLOATING_TYPES(
      kpl_input.scalar_type(), "kplane_forward_cuda", [&] {
      if (canUse32BitIndexMath(kpl_input) && canUse32BitIndexMath(grid) &&
          canUse32BitIndexMath(output)) {

  #if INFO_SHMEM
        printf("shmem_size: %d, kpl/out/wmma = (%d, %d, %d)\n", shmem_size, num_kpl, num_out, num_wmma);
  #endif

        cudaFuncSetAttribute(kplane_forward_kernel<scalar_t, int, Interpolation_E, Padding_E, Align_E, Fushion_E>, cudaFuncAttributeMaxDynamicSharedMemorySize , shmem_size);
        // cudaFuncSetCacheConfig(kplane_mlp_forward_kernel<scalar_t, int>, cudaFuncCachePreferShared);
        // cudaFuncSetCacheConfig(kplane_mlp_forward_kernel<scalar_t, int>, cudaFuncCachePreferL1);
        kplane_forward_kernel<scalar_t, int, Interpolation_E, Padding_E, Align_E, Fushion_E>
          <<<blocks, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
            #if USE_CONST_TI > 0
            getTensorInfo<const scalar_t, int>(kpl_input),
            getTensorInfo<const scalar_t, int>(grid),
            getTensorInfo<const scalar_t, int>(skip),
            #else
            getTensorInfo<scalar_t, int>(kpl_input),
            getTensorInfo<scalar_t, int>(grid),
            getTensorInfo<scalar_t, int>(skip),
            #endif

          #if USE_TEMPLATE_ARGS
            getTensorInfo<scalar_t, int>(output)
          #else
            getTensorInfo<scalar_t, int>(output),
            static_cast<GridSamplerInterpolation>(interpolation_mode),
            static_cast<GridSamplerPadding>(padding_mode),
            align_corners,
            static_cast<FeatFusion>(feature_fusion)
          #endif

            );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {

  #if INFO_SHMEM
        printf("shmem_size: %d, kpl/out/wmma = (%d, %d, %d)\n", shmem_size, num_kpl, num_out, num_wmma);
  #endif

        cudaFuncSetAttribute(kplane_forward_kernel<scalar_t, int64_t, Interpolation_E, Padding_E, Align_E, Fushion_E>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
        // cudaFuncSetCacheConfig(kplane_mlp_forward_kernel<scalar_t, int64_t>, cudaFuncCachePreferShared);
        // cudaFuncSetCacheConfig(kplane_mlp_forward_kernel<scalar_t, int64_t>, cudaFuncCachePreferL1);
        kplane_forward_kernel<scalar_t, int64_t, Interpolation_E, Padding_E, Align_E, Fushion_E>
          <<<blocks, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(

            #if USE_CONST_TI > 0
            getTensorInfo<const scalar_t, int64_t>(kpl_input),
            getTensorInfo<const scalar_t, int64_t>(grid),
            getTensorInfo<const scalar_t, int64_t>(skip),
            #else
            getTensorInfo<scalar_t, int64_t>(kpl_input),
            getTensorInfo<scalar_t, int64_t>(grid),
            getTensorInfo<scalar_t, int64_t>(skip),
            #endif

          #if USE_TEMPLATE_ARGS
            getTensorInfo<scalar_t, int64_t>(output)

          #else
            getTensorInfo<scalar_t, int64_t>(output),

            static_cast<GridSamplerInterpolation>(interpolation_mode),
            static_cast<GridSamplerPadding>(padding_mode),
            align_corners,
            static_cast<FeatFusion>(feature_fusion)
          #endif

            );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    });

    }));
    }));
    }));
    }));

  }

  return output;
}

std::vector<torch::Tensor> kplane_backward_cuda(
// void kplane_mlp_backward_cuda(torch::Tensor &grad_kpl, torch::Tensor &grad_mlp, torch::Tensor &grad_grid, torch::Tensor &grad_skip,
    const torch::Tensor &grad_output,

    const torch::Tensor &kpl_param, 
    const torch::Tensor &grid,      const torch::Tensor &skip,      

    int64_t interpolation_mode, int64_t padding_mode, bool align_corners, int64_t feature_fusion,
    
    std::array<bool, 3> output_mask
) {

  // auto kpl_input = at::permute(kpl_param, {0, 3, 4, 1, 2}).contiguous();
  // auto kpl_input = kpl_param;
  auto kpl_input = kpl_param;

  auto N = kpl_input.size(0);
  auto K = kpl_input.size(3);
  auto C = kpl_input.size(4);

  auto skip_C = skip.size(3);

  auto H = grid.size(1); // output H
  auto W = grid.size(2); // output W
  int64_t count = N * H * W;

  const int64_t kpl_C = C + skip_C;
  const int64_t out_C = kpl_C;

  const int64_t max_C = kpl_C;

  bool kpl_requires_grad  = output_mask[0];
  bool grid_requires_grad = output_mask[1];
  bool skip_requires_grad = output_mask[2];

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

  constexpr unsigned align_float = ALIGN_BYTES / sizeof(float);

  if (count > 0) {
    constexpr unsigned THREAD_Y = 1; // MLP_CHUNK_SIZE/4

    // dim3 blocks  = {(H*W-1)/MLP_CHUNK_SIZE + 1, N};
    dim3 blocks  = {H*W, N};
    dim3 threads = {32, THREAD_Y};

    int temp_C   = max_C%align_float ? max_C : (max_C + align_float - max_C%align_float);
    int num_kpl  = sizeof(float) * (THREAD_Y*(K*C));

    unsigned shmem_size = num_kpl;

    AT_DISPATCH_FLOATING_TYPES(
      kpl_input.scalar_type(), "kplane_backward_cuda", [&] {
      if (canUse32BitIndexMath(kpl_input) && canUse32BitIndexMath(grid) &&
          canUse32BitIndexMath(grad_output)) {
        
        // cudaFuncSetAttribute(kplane_mlp_backward_kernel<scalar_t, int>, cudaFuncAttributeMaxDynamicSharedMemorySize , shmem_size);
        // cudaFuncSetCacheConfig(kplane_mlp_backward_kernel<scalar_t, int>, cudaFuncCachePreferShared);
        kplane_backward_kernel<scalar_t>
          <<<blocks, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(

            #if USE_CONST_TI > 0
            getTensorInfo<const scalar_t, int>(grad_output),
            getTensorInfo<const scalar_t, int>(kpl_input),
            getTensorInfo<const scalar_t, int>(grid),
            getTensorInfo<const scalar_t, int>(skip),
            #else
            getTensorInfo<scalar_t, int>(grad_output),
            getTensorInfo<scalar_t, int>(kpl_input),
            getTensorInfo<scalar_t, int>(grid),
            getTensorInfo<scalar_t, int>(skip),
            #endif

            kpl_requires_grad  ? getTensorInfo<scalar_t, int>(grad_kpl)  : TensorInfo<scalar_t, int>(),
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

            kpl_requires_grad  ? static_cast<int>(grad_kpl.numel() ) : 0,
            grid_requires_grad ? static_cast<int>(grad_grid.numel()) : 0,
            skip_requires_grad ? static_cast<int>(grad_skip.numel()) : 0
            );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {

        // cudaFuncSetAttribute(kplane_mlp_backward_kernel<scalar_t, int>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
        // cudaFuncSetCacheConfig(kplane_mlp_backward_kernel<scalar_t, int64_t>, cudaFuncCachePreferShared);
        kplane_backward_kernel<scalar_t>
          <<<blocks, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(

            #if USE_CONST_TI > 0
            getTensorInfo<const scalar_t, int64_t>(grad_output),
            getTensorInfo<const scalar_t, int64_t>(kpl_input),
            getTensorInfo<const scalar_t, int64_t>(grid),
            getTensorInfo<const scalar_t, int64_t>(skip),
            #else
            getTensorInfo<scalar_t, int64_t>(grad_output),
            getTensorInfo<scalar_t, int64_t>(kpl_input),
            getTensorInfo<scalar_t, int64_t>(grid),
            getTensorInfo<scalar_t, int64_t>(skip),
            #endif

            kpl_requires_grad  ? getTensorInfo<scalar_t, int64_t>(grad_kpl)  : TensorInfo<scalar_t, int64_t>(),
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
            
            kpl_requires_grad  ? grad_kpl.numel()  : 0,
            grid_requires_grad ? grad_grid.numel() : 0,
            skip_requires_grad ? grad_skip.numel() : 0
            );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    });
  }

  // auto kpl_input = at::permute(kpl_param, {0, 3, 4, 1, 2}).contiguous();

  return {grad_kpl, grad_grid, grad_skip};
}

}  // namespace at::native