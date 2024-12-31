#ifndef KPLANE_MLP_TYPE_CUH
#define KPLANE_MLP_TYPE_CUH

template<typename scalar_t>
struct CudaType{
    // static const scalar_t zero_v {static_cast<scalar_t>(0)};
    __inline__ __device__ static scalar_t from_float(const float a){return a;};
    __inline__ __device__ static float    to_float(const scalar_t a){return a;};
    __inline__ __device__ static scalar_t get_zero_v(){return static_cast<scalar_t>(0);};

    // template<typename T>
    // __inline__ __device__ static scalar_t from(const T a){return a;};
};

template<>
struct CudaType<half>{
    // static const half zero_v {__int2half_rz(0)};
    __inline__ __device__ static half from_float(const float a){return __float2half_rz(a);};
    __inline__ __device__ static float to_float(const half a){return __half2float(a);};
    __inline__ __device__ static half get_zero_v(){return __int2half_rz(0);};

    // template<typename T>
    // __inline__ __device__ static half from(const T a);
    // template<>
    // __inline__ __device__ static half from(const float a){return from_float(a);};
};

#endif