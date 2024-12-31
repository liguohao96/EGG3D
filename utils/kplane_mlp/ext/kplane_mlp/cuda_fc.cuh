#ifndef KPLANE_MLP_FC_CUH
#define KPLANE_MLP_FC_CUH

#include "common.h"
#include "cuda_type.cuh"

// WMMA (Tensor Core)
#include <mma.h>
using namespace nvcuda;

template <typename index_t, index_t WARP_CHUNK, index_t WMMA_M, index_t WMMA_N, index_t WMMA_K>
struct FullyConnectedLayer{

  __inline__ __device__ index_t get_output_stride (const index_t C){ // output stride align to 8 float
    return (C&7 == 0) ? (C) : (C + (8-C&7));                             // C%8 == 0 ? C : C + (8-C%8)
  };

  __inline__ __device__ void warp_fc(
    const float* __restrict__ mat_a,  // WARP_CHUNK x ci row_major
    const float* __restrict__ mat_b,  //         ci x co col_major
          float*              mat_c,  // WARP_CHUNK x co row_major
    
    const index_t rs_a, const index_t cs_b, const index_t rs_c,

    const index_t ci, const index_t co
  ){
  };

  __inline__ __device__ void fast_warp_fc(
    const half* __restrict__ mat_a,  // WARP_CHUNK x ci row_major
    const half* __restrict__ mat_b,  //         ci x co col_major
          float*             mat_c,  // WARP_CHUNK x co row_major
    
    const index_t rs_a, const index_t cs_b, const index_t rs_c,

    const index_t ci, const index_t co
  );

  template<typename scalar_t>
  __inline__ __device__ void slow_warp_fc(
    const scalar_t* __restrict__ mat_a,  // WARP_CHUNK x ci row_major
    const scalar_t* __restrict__ mat_b,  //         ci x co col_major
          float*                 mat_c,  // WARP_CHUNK x co row_major
    
    const index_t rs_a, const index_t cs_b, const index_t rs_c,

    const index_t ci, const index_t co
  );

  // template <typename scalar_t>
  // __inline__ __device__ void FullyConnectedLayer<index_t, WARP_CHUNK, WMMA_M, WMMA_N, WMMA_K>::slow_warp_fc<half>(
  //   const half* __restrict__ mat_a,  // WARP_CHUNK x ci row_major
  //   const half* __restrict__ mat_b,  //         ci x co col_major
  //         float*             mat_c,  // WARP_CHUNK x co row_major
    
  //   const index_t rs_a, const index_t cs_b, const index_t rs_c,

  //   const index_t ci, const index_t co
  // ){
  //   constexpr index_t tile_K = 2;  // h
  //   constexpr index_t tile_I = 16;  // w

  //   constexpr index_t tile_C = tile_I;  // w

  //   const index_t tid          = threadIdx.x;
  //   const index_t cid          = threadIdx.y;
  //   // const index_t chunk_stride = blockDim.y;
  //   constexpr index_t chunk_stride = WARP_CHUNK/4;

  //   // option: A 0.49x
  //   #pragma unroll
  //   for (index_t m=tid/tile_I*chunk_stride; m<WARP_CHUNK; m+=tile_K*chunk_stride){ // [0, 0, 0, 0, 0, 0, 0, 0, ..., 3, 3, 3, 3, 3, 3, 3, 3] -> [4, 4, 4, 4, 4, 4, 4, 4, ..., 7, 7, 7, 7, 7, 7, 7, 7]
  //     for (index_t o=tid&(tile_I-1); o<co; o+=tile_I){         // [0, 1, 2, 3, 4, 5, 6, 7, ..., 0, 1, 2, 3, 4, 5, 6, 7] -> [8, 9, ...]
  //       half out = __int2half_rz(0);
  //       #pragma unroll
  //       for (index_t i=0; i<ci; ++i){
  //         out = __hfma(mat_a[m*rs_a + i], mat_b[i + o*cs_b], out); // mat_a[m,i] * mat_b[i,o]
  //       }
  //       mat_c[m*rs_c + o] = __half2float(out);  // mat_c[m,o] = 
  //     }
  //   }
  // };

};

template <typename index_t, index_t WARP_CHUNK, index_t WMMA_M, index_t WMMA_N, index_t WMMA_K>
__inline__ __device__ void FullyConnectedLayer<index_t, WARP_CHUNK, WMMA_M, WMMA_N, WMMA_K>::fast_warp_fc(
  const half * __restrict__ mat_a,  // WARP_CHUNK x ci row_major
  const half * __restrict__ mat_b,  //         co x ci col_major
        float*              mat_c,  // WARP_CHUNK x co row_major
  
  const index_t rs_a, const index_t cs_b, const index_t rs_c,

  const index_t ci, const index_t co
){
  const index_t cid          = threadIdx.y;
  // const index_t chunk_stride = blockDim.y;

  constexpr index_t chunk_stride = WARP_CHUNK/4;

  wmma::fragment<wmma::matrix_a,    WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b,    WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>                 c_frag;
  // wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>                 d_frag;

  for (index_t o=cid*WMMA_N; o<co; o+=WMMA_N*chunk_stride){

    // Initialize the output to zero
    wmma::fill_fragment(c_frag, 0.0f); // bias

    float* c_ptr = mat_c + o;
    for (index_t k=0; k<ci; k+=WMMA_K){

      const half* __restrict__ a_ptr = mat_a + k          ;
      const half* __restrict__ b_ptr = mat_b + k + o*cs_b ;

      // Load the inputs
      wmma::load_matrix_sync(a_frag, a_ptr, rs_a);
      wmma::load_matrix_sync(b_frag, b_ptr, cs_b);

      // Perform the matrix multiplication
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag); /* satf=true */ // clamp NaN and Inf

      // Store the output
    }

    wmma::store_matrix_sync(c_ptr, c_frag, rs_c, wmma::mem_row_major);
  }

};

template <typename index_t, index_t WARP_CHUNK, index_t WMMA_M, index_t WMMA_N, index_t WMMA_K>
template <typename scalar_t>
__inline__ __device__ void FullyConnectedLayer<index_t, WARP_CHUNK, WMMA_M, WMMA_N, WMMA_K>::slow_warp_fc(
  const scalar_t* __restrict__ mat_a,  // WARP_CHUNK x ci row_major
  const scalar_t* __restrict__ mat_b,  //         ci x co col_major
        float*                 mat_c,  // WARP_CHUNK x co row_major
  
  const index_t rs_a, const index_t cs_b, const index_t rs_c,

  const index_t ci, const index_t co
){
  const unsigned tid = threadIdx.x;
  const unsigned cid = threadIdx.y;

  // constexpr index_t tile_K = 2;  // h
  // constexpr index_t tile_I = 16;  // w

 // constexpr index_t tile_C = tile_I;  // w

  // assert tile_K*tile_I == warp_size == 32

  // const index_t chunk_stride = blockDim.y;
  // constexpr index_t chunk_stride = WARP_CHUNK/4;

  // option: A easy 0.43x
  /*
    hard
    is_close:  True mean: 0.0 max: 0.0
    [      raw      ] 16.135ms | 1.00x
    [  raw_no_grad  ] 16.056ms | 1.00x
    [  raw_compile  ] 16.135ms | 1.00x
    [  fuse_kplane  ] 53.318ms | 0.30x
  */
  // #pragma unroll
  // for (index_t m=tid/tile_I+cid*tile_K; m<WARP_CHUNK; m+=tile_K*chunk_stride){ // [0, 0, 0, 0, 0, 0, 0, 0, ..., 3, 3, 3, 3, 3, 3, 3, 3] -> [4, 4, 4, 4, 4, 4, 4, 4, ..., 7, 7, 7, 7, 7, 7, 7, 7]
  //   for (index_t o=tid&(tile_I-1); o<co; o+=tile_I){         // [0, 1, 2, 3, 4, 5, 6, 7, ..., 0, 1, 2, 3, 4, 5, 6, 7] -> [8, 9, ...]
  //     float out = static_cast<float>(0);
  //     #pragma unroll
  //     for (index_t i=0; i<ci; ++i){
  //       // out += mat_a[m*rs_a + i] * mat_b[i + o*cs_b]; // mat_a[m,i] * mat_b[i,o]
  //       out = fmaf(mat_a[m*rs_a + i], mat_b[i + o*cs_b], out); // mat_a[m,i] * mat_b[i,o]
  //     }
  //     mat_c[m*rs_c + o] = out;  // mat_c[m,o] = 
  //   }
  // }

  // option: B
  /*
    hard
    is_close:  True mean: 3.053541774988844e-08 max: 5.960464477539062e-07
    /home/vc-master/anaconda3/envs/unigen/lib/python3.10/site-packages/torch/_inductor/compile_fx.py:124: UserWarning: TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled. Consider setting `torch.set_float32_matmul_precision('high')` for better performance.
      warnings.warn(
    [      raw      ] 16.199ms | 1.00x
    [  raw_no_grad  ] 16.069ms | 1.01x
    [  raw_compile  ] 16.133ms | 1.00x
    [  fuse_kplane  ] 42.000ms | 0.39x
  */
  const index_t mi_index[4] = {cid<<2, (cid<<2)|1, (cid<<2)|2, (cid<<2)|3};
  // const index_t mi_index[4] = {cid*4, cid*4+1, cid*4+2, cid*4+3};

  constexpr unsigned logN = 3;
  constexpr unsigned N    = 1<<logN;         // out channel N
  constexpr unsigned K    = 32>>logN;        // continuous
  constexpr unsigned FULL_MASK = 0xffffffff;
  const unsigned mask = FULL_MASK;
  const bool in_range = true;
  
  // const unsigned mod = tid%K;
  const unsigned mod = tid&(K-1);
  const unsigned div = tid/K;
  const bool write = mod == 0;

  // if (blockIdx.x + blockIdx.y == 0 && threadIdx.y == 0 && tid&(K-1) != tid%K)
  //   printf("%d %d %d %d %d %d %d\n", logN, N, K, (tid/K), tid, tid&(K-1), tid%K);

  const float zero = CudaType<float>::get_zero_v();

  // #pragma unroll
  for(index_t ob=0; ob<co; ob+=N){
  // for(index_t o=0; o<co; o+=1){

    index_t o = ob + div;
    const bool in_range = o < co;

    float stride_buffer[4] = {zero, zero, zero, zero};

    if (in_range)
    for(index_t i=mod; i<ci; i+=K){
    // for(index_t i=tid; i<ci; i+=32){
      float b_val = mat_b[i + o*cs_b];

      {
        stride_buffer[0] = fmaf(mat_a[ mi_index[0]*rs_a + i ], b_val, stride_buffer[0]);
        stride_buffer[1] = fmaf(mat_a[ mi_index[1]*rs_a + i ], b_val, stride_buffer[1]);
        stride_buffer[2] = fmaf(mat_a[ mi_index[2]*rs_a + i ], b_val, stride_buffer[2]);
        stride_buffer[3] = fmaf(mat_a[ mi_index[3]*rs_a + i ], b_val, stride_buffer[3]);
      }

    }

    unsigned mask = __ballot_sync(FULL_MASK, in_range);

    #pragma unroll
    for(index_t i=0; i<4; ++i){
      float   out = stride_buffer[i];
      index_t idx = mi_index[i];

      // for(unsigned s=32>>(logN+1); s>0; s>>1)
      //   out += __shfl_down_sync(mask, out, s);

      // out += __shfl_down_sync(mask, out, 16);
      // out += __shfl_down_sync(mask, out, 8);
      // out += __shfl_down_sync(mask, out, 4);
      out += __shfl_down_sync(mask, out, 2);
      out += __shfl_down_sync(mask, out, 1);

      if (in_range && write)
      // if (in_range && tid == 0)
        mat_c[ idx*rs_c + o ] = out;

    }
  }

};


#endif