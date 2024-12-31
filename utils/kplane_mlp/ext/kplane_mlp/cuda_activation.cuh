#ifndef KPLANE_MLP_ACT_CUH
#define KPLANE_MLP_ACT_CUH

#include "common.h"
#include "cuda_type.cuh"

template <typename index_t, index_t WARP_CHUNK>
struct ActivationFunction{

  Activation act;

  template <typename scalar_t>
  __inline__ __device__ void warp_act(
    const float* __restrict__ mat_out,   // row major
          scalar_t*           act_out,   // row major

    const index_t rs_m, const index_t rs_a, // row stride
    const index_t rn_m                      // row size
  ){
    // constexpr scalar_t zero = CudaType<scalar_t>::get_zero_v();
    // constexpr scalar_t zero = CudaType<scalar_t>::zero_v;
    const scalar_t zero = CudaType<scalar_t>::get_zero_v();

    const index_t tid = threadIdx.x;
    const index_t cid = threadIdx.y;

    // const index_t chunk_stride = blockDim.y;
    constexpr index_t chunk_stride = WARP_CHUNK/4;

    if (act == Activation::Softplus){
      float beta      = static_cast<float>(10); // same as tiny-cuda-nn https://github.com/NVlabs/tiny-cuda-nn/blob/b3473c81396fe927293bdfd5a6be32df8769927c/include/tiny-cuda-nn/common_device.h#L100
      float threshold = static_cast<float>(20);

      #pragma unroll
      for (index_t m=cid; m<WARP_CHUNK; m+=chunk_stride){
        for (index_t i=tid; i<rn_m; i+=32){
          float x      = mat_out[m*rs_m + i];
          float beta_x = beta * x;
          float act_vf = beta_x > threshold ? x : (log1pf(expf(beta_x))/beta);
        
          act_out[m*rs_a + i] = CudaType<scalar_t>::from_float(act_vf);
        }
      }

    } else if (act == Activation::ReLU){
      float clamp = static_cast<float>(0);

      #pragma unroll
      for (index_t m=cid; m<WARP_CHUNK; m+=chunk_stride){
        for (index_t i=tid; i<rn_m; i+=32){
          float act_vf = fmaxf(mat_out[m*rs_m + i], clamp);

          act_out[m*rs_a + i] = CudaType<scalar_t>::from_float(act_vf);
        }
      }
    } else {

      // Copy
      #pragma unroll
      for (index_t m=cid; m<WARP_CHUNK; m+=chunk_stride){
        for (index_t i=tid; i<rn_m; i+=32){
          float act_vf = mat_out[m*rs_m + i];

          act_out[m*rs_a + i] = CudaType<scalar_t>::from_float(act_vf);
        }
      }

    }

    // // pad zero
    // #pragma unroll
    // for (index_t m=cid; m<WARP_CHUNK; m+=chunk_stride){
    //   for (index_t i=rn_m+tid; i<rs_a; i+=32){
    //     act_out[m*rs_a + i] = zero;
    //   }
    // }

  };

  // template <typename scalar_t>
  // __inline__ __device__ void warp_act<half>(
  //   const float* __restrict__ mat_out,   // row major
  //          half*              act_out,   // row major

  //   const index_t rs_m, const index_t rs_a, // row stride
  //   const index_t rn_m                      // row size
  // ){
  //   const half half_z = __int2half_rz(0);

  //   const index_t tid = threadIdx.x;
  //   const index_t cid = threadIdx.y;

  //   // const index_t chunk_stride = blockDim.y;
  //   constexpr index_t chunk_stride = WARP_CHUNK/4;

  //   if (act == Activation::Softplus){
  //     float beta      = static_cast<float>(10); // same as tiny-cuda-nn https://github.com/NVlabs/tiny-cuda-nn/blob/b3473c81396fe927293bdfd5a6be32df8769927c/include/tiny-cuda-nn/common_device.h#L100
  //     float threshold = static_cast<float>(20);

  //     #pragma unroll
  //     for (index_t m=cid; m<WARP_CHUNK; m+=chunk_stride){
  //       for (index_t i=tid; i<rn_m; i+=32){
  //         float x      = mat_out[m*rs_m + i];
  //         float beta_x = beta * x;
  //         float act_vf = beta_x > threshold ? x : (log1pf(expf(beta_x))/beta);
        
  //         act_out[m*rs_a + i] = __float2half_rz(act_vf);
  //       }
  //     }

  //   } else if (act == Activation::ReLU){
  //     float clamp = static_cast<float>(0);

  //     #pragma unroll
  //     for (index_t m=cid; m<WARP_CHUNK; m+=chunk_stride){
  //       for (index_t i=tid; i<rn_m; i+=32){
  //         float act_vf = fmaxf(mat_out[m*rs_m + i], clamp);

  //         act_out[m*rs_a + i] = __float2half_rz(act_vf);
  //       }
  //     }
  //   } else {

  //     // Copy
  //     #pragma unroll
  //     for (index_t m=cid; m<WARP_CHUNK; m+=chunk_stride){
  //       for (index_t i=tid; i<rn_m; i+=32){
  //         float act_vf = mat_out[m*rs_m + i];

  //         act_out[m*rs_a + i] = __float2half_rz(act_vf);
  //       }
  //     }

  //   }

  //   // pad zero
  //   #pragma unroll
  //   for (index_t m=cid; m<WARP_CHUNK; m+=chunk_stride){
  //     for (index_t i=rn_m+tid; i<rs_a; i+=32){
  //       act_out[m*rs_a + i] = half_z;
  //     }
  //   }

  // };

};



template <typename scalar_t, typename index_t, index_t WARP_CHUNK>
__inline__ __device__ void warp_inplace_act_fn(
  float* mat_out,   // row major

  const index_t r_s, const index_t r_n, // row stride, row size
  const Activation act
){

  const index_t tid = threadIdx.x;
  constexpr float zero = static_cast<float>(0);

  if (act == Activation::Softplus){
    float beta      = static_cast<float>(10); // same as tiny-cuda-nn https://github.com/NVlabs/tiny-cuda-nn/blob/b3473c81396fe927293bdfd5a6be32df8769927c/include/tiny-cuda-nn/common_device.h#L100
    float threshold = static_cast<float>(20);

    #pragma unroll
    for (index_t m=0; m<WARP_CHUNK; m+=1){
      for (index_t i=tid; i<r_n; i+=32){
        float x      = mat_out[m*r_s + i];
        float beta_x = beta * x;
        float act_vf = beta_x > threshold ? x : (log1pf(expf(beta_x))/beta);
        
        mat_out[m*r_s + i] = act_vf;
      }

      for (int i=tid+r_n; i<r_s; i+=32)
        mat_out[m*r_s + i] = zero;
    }

  } else if (act == Activation::ReLU){
    float clamp = static_cast<float>(0);

    #pragma unroll
    for (index_t m=0; m<WARP_CHUNK; m+=1){
      for (int i=tid; i<r_n; i+=32){
        float act_vf = fmaxf(mat_out[m*r_s + i], clamp);

        mat_out[m*r_s + i] = act_vf;
      }

      for (int i=tid+r_n; i<r_s; i+=32)
        mat_out[m*r_s + i] = zero;

    }
  }
}
#endif