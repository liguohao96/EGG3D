#include <torch/types.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>
#include "utils.h"

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
namespace mesh2sd{
// start of namespace

// #define max(a,b) \
//    ({ __typeof__ (a) _a = (a); \
//        __typeof__ (b) _b = (b); \
//      _a > _b ? _a : _b; })

// #define min(a,b) \
//    ({ __typeof__ (a) _a = (a); \
//        __typeof__ (b) _b = (b); \
//      _a < _b ? _a : _b; })

#define min3(a,b,c) (min(min(a,b), c))
#define max3(a,b,c) (max(max(a,b), c))

#define USE_SHARED
// #define USE_ACCESS
#define USE_BLOCKZ
#define TRI_STEP 32

#define ALGO_V2 1
// #define CUDA_DEBUG

#define INFINITY 99999

template <typename scalar_t, typename index_t>
__global__ void sd2mesh_forward_cuda_impl(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> faces, // b, nf, 3, 3
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> query, // b, n, 3

    torch::PackedTensorAccessor32<scalar_t,  2, torch::RestrictPtrTraits>      d_buf, // b, n
    torch::PackedTensorAccessor32<index_t,   2, torch::RestrictPtrTraits>      i_buf, // b, n
    torch::PackedTensorAccessor32<scalar_t,  3, torch::RestrictPtrTraits>      w_buf  // b, n, 3
){
    const int BATCH_SIZE = faces.size(0);
    const int NUM_FACES  = faces.size(1);
    const int NUM_QUERY  = query.size(1);

    // scalar_t EPS = 1e-5;
    const int bx  = blockIdx.x,  by = blockIdx.y,  bz = blockIdx.z;
    const int tx  = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;

    const int bi = bz;
    const int qi = bx;

    extern __shared__ char shared_mem[];
    const int s_num_b   = (blockDim.x*5) * sizeof(scalar_t);
    const int s_pad_b   = s_num_b % sizeof(index_t);
    scalar_t* s_s_dist  = (scalar_t*)      &shared_mem[0];
    scalar_t* s_u_dist  = (scalar_t*)      &shared_mem[sizeof(scalar_t)*blockDim.x];
    vec3<scalar_t>* s_w3= (vec3<scalar_t>*)&shared_mem[sizeof(scalar_t)*blockDim.x*2];
    index_t*       s_tid= (index_t*)       &shared_mem[sizeof(scalar_t)*blockDim.x*5+s_pad_b];

    vec3<scalar_t> xyz;
    vec3<scalar_t> local_v33[3];

    xyz.x = query[bi][qi][0];
    xyz.y = query[bi][qi][1];
    xyz.z = query[bi][qi][2];

    scalar_t s_dist = static_cast<scalar_t>(INFINITY);
    scalar_t u_dist = static_cast<scalar_t>(INFINITY);
    index_t  tid = -1;
    vec3<scalar_t> w3;

    vec4<scalar_t> out;
    // scalar_t       s_dist;
#ifdef CUDA_DEBUG
    printf("query xyz (%f, %f, %f)\n", xyz.x, xyz.y, xyz.z);
#endif

    for(int fi=0; fi<NUM_FACES; fi+=blockDim.x){
        bool in_range = fi+tx < NUM_FACES;

        if (in_range){
            local_v33[0].x = faces[bi][fi+tx][0][0];
            local_v33[0].y = faces[bi][fi+tx][0][1];
            local_v33[0].z = faces[bi][fi+tx][0][2];
            local_v33[1].x = faces[bi][fi+tx][1][0];
            local_v33[1].y = faces[bi][fi+tx][1][1];
            local_v33[1].z = faces[bi][fi+tx][1][2];
            local_v33[2].x = faces[bi][fi+tx][2][0];
            local_v33[2].y = faces[bi][fi+tx][2][1];
            local_v33[2].z = faces[bi][fi+tx][2][2];
        }
        __syncthreads();

        point_triangle_CP<scalar_t>(xyz, local_v33[0], local_v33[1], local_v33[2], out);
        
        scalar_t abs_d = fabs(out.w);

        bool is_closer = abs_d < u_dist - EPS;
        bool is_equal  = fabs(abs_d - u_dist) < EPS;
        bool is_signgt = sign(out.w) > sign(s_dist);   // new tirangle got + distance, origin is - distance

        bool change = is_closer || ( is_equal && is_signgt );

        if ( in_range && change ){
            s_dist = out.w;
            u_dist = abs_d;
            w3.x   = out.x;
            w3.y   = out.y;
            w3.z   = out.x;
            tid    = fi+tx;
        }
    }

    s_s_dist[tx] = s_dist;
    s_u_dist[tx] = u_dist;
    s_w3[tx]     = w3;
    s_tid[tx]    = tid;
    __syncthreads();

    if (tx == 0){
        for(int i=1;i<blockDim.x;++i){
            scalar_t abs_d = s_u_dist[i];

            bool in_range  = s_tid[i] >= 0;

            bool is_closer = abs_d < u_dist - EPS;
            bool is_equal  = fabs(abs_d - u_dist) < EPS;
            bool is_signgt = sign(s_s_dist[i]) > sign(s_dist);   // new tirangle got + distance, origin is - distance

            bool change = is_closer || ( is_equal && is_signgt );

            if ( in_range && change ){
                s_dist = s_s_dist[i];
                u_dist = abs_d;
                w3.x   = s_w3[i].x;
                w3.y   = s_w3[i].y;
                w3.z   = s_w3[i].x;
                tid    = s_tid[i];
            }
        }

        d_buf[bi][qi]    = s_dist;
        i_buf[bi][qi]    = tid;
        w_buf[bi][qi][0] = w3.x;
        w_buf[bi][qi][1] = w3.y;
        w_buf[bi][qi][2] = w3.z;
    }
}

template <typename scalar_t, typename index_t>
__global__ void sd2mesh_forward_cuda_impl_v2(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> faces, // b, nf, 3, 3
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> query, // b, n, 3

    torch::PackedTensorAccessor32<scalar_t,  2, torch::RestrictPtrTraits>      d_buf, // b, n
    torch::PackedTensorAccessor32<index_t,   2, torch::RestrictPtrTraits>      i_buf, // b, n
    torch::PackedTensorAccessor32<scalar_t,  3, torch::RestrictPtrTraits>      w_buf  // b, n, 3
){
    const int BATCH_SIZE = faces.size(0);
    const int NUM_FACES  = faces.size(1);
    const int NUM_QUERY  = query.size(1);

    // scalar_t EPS = 1e-5;
    const int bx  = blockIdx.x,  by = blockIdx.y,  bz = blockIdx.z;
    const int tx  = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;

    const int bi = bz;
    const int qi = bx*blockDim.x + tx;

    extern __shared__ char shared_mem[];
    vec3<scalar_t> *s_v = (vec3<scalar_t>(*))&shared_mem[0];
    vec7<scalar_t> *s_p = (vec7<scalar_t>(*))&shared_mem[blockDim.x*3*3*sizeof(scalar_t)];

    bool qi_in_range = qi < NUM_QUERY;
    vec3<scalar_t> xyz;

    if (qi_in_range){
        xyz.x = query[bi][qi][0];
        xyz.y = query[bi][qi][1];
        xyz.z = query[bi][qi][2];
    }

    scalar_t s_dist = static_cast<scalar_t>(INFINITY);
    scalar_t u_dist = static_cast<scalar_t>(INFINITY);
    int tid = -1;
    vec3<scalar_t> w3;
    vec4<scalar_t> out;
    vec3<scalar_t> v0, v1, v2;

#ifdef CUDA_DEBUG
    printf("query xyz %d (%f, %f, %f)\n", qi, xyz.x, xyz.y, xyz.z);
#endif

    for(int fi=0; fi<NUM_FACES; fi+=blockDim.x){
        bool in_range = fi+tx < NUM_FACES;

        if (in_range){
            v0.x = faces[bi][fi+tx][0][0];
            v0.y = faces[bi][fi+tx][0][1];
            v0.z = faces[bi][fi+tx][0][2];
            v1.x = faces[bi][fi+tx][1][0];
            v1.y = faces[bi][fi+tx][1][1];
            v1.z = faces[bi][fi+tx][1][2];
            v2.x = faces[bi][fi+tx][2][0];
            v2.y = faces[bi][fi+tx][2][1];
            v2.z = faces[bi][fi+tx][2][2];

            scalar_t d00 = (v1 - v0).dot(v1 - v0);
            scalar_t d01 = (v1 - v0).dot(v2 - v0);
            scalar_t d11 = (v2 - v0).dot(v2 - v0);

            vec3<scalar_t> normal = (v1 - v0).cross(v2 - v0);
            scalar_t invDet       = normal.rnorm();
            normal = normal * invDet;

            s_v[tx*3 + 0] = v0;
            s_v[tx*3 + 1] = v1;
            s_v[tx*3 + 2] = v2;

            s_p[tx].x = normal.x;
            s_p[tx].y = normal.y;
            s_p[tx].z = normal.z;
            s_p[tx].w = 1 / (d00 * d11 - d01 * d01);
            // s_p[tx].w = invDet*invDet;
            // s_p[tx].w = invDet;
            s_p[tx].a = d00;
            s_p[tx].b = d01;
            s_p[tx].c = d11;
        }
        __syncthreads();

        const int lt_len = min(blockDim.x, NUM_FACES-fi);
#ifdef CUDA_DEBUG
        printf("face ID %d/%d, lt_len %d\n", fi, NUM_FACES, lt_len);
#endif

        for(int lti=0; lti<lt_len; ++lti){
            // point_triangle_CP<scalar_t>(xyz, s_v0[lti], s_v1[lti], s_v2[lti], out);
            vec3<scalar_t> v0 = s_v[lti*3+0];
            vec3<scalar_t> v1 = s_v[lti*3+1];
            vec3<scalar_t> v2 = s_v[lti*3+2];
            vec7<scalar_t> pre= s_p[lti];
            point_triangle_CP<scalar_t>(xyz, v0, v1, v2, pre, out);

            scalar_t abs_d = fabsf(out.w);

            bool is_closer = abs_d < u_dist - EPS;
            bool is_equal  = fabsf(abs_d - u_dist) < EPS;
            bool is_signgt = sign(out.w) > sign(s_dist);   // new tirangle got + distance, origin is - distance

            bool change = is_closer || ( is_equal && is_signgt );
            if ( change ){
                s_dist = out.w;
                u_dist = abs_d;
                w3.x   = out.x;
                w3.y   = out.y;
                w3.z   = out.x;
                tid    = fi+lti;
            }
        }
    }

    if (qi_in_range){
        d_buf[bi][qi]    = s_dist;
        i_buf[bi][qi]    = tid;
        w_buf[bi][qi][0] = w3.x;
        w_buf[bi][qi][1] = w3.y;
        w_buf[bi][qi][2] = w3.z;
    }
}

std::vector<torch::Tensor> sd_query_forward_cuda(
    torch::Tensor faces,
    torch::Tensor query
){
    // https://github.com/iamyoukou/sdf3d
    const unsigned int  batch_size = faces.size(0);
    const unsigned int  num_tri    = faces.size(1);
    const unsigned int  NQ         = query.size(1);

    auto tri_id = torch::empty({batch_size, NQ},    faces.options().dtype(torch::kInt64));
    auto s_dist = torch::empty({batch_size, NQ},    faces.options());
    auto weight = torch::empty({batch_size, NQ, 3}, faces.options());

#ifdef ALGO_V1
    const unsigned int THREAD_W = 32;
    const unsigned int THREAD_H = 1;

    const dim3 block_conf{THREAD_W, THREAD_H};
    const dim3 grid_conf {NQ, 1, batch_size};
#elif ALGO_V2
    const unsigned int THREAD_W = 32;
    const unsigned int THREAD_H = 1;
    const unsigned int BLOCK_W  = (NQ-1)/THREAD_W+1; // (NQ%THREAD_W == 0) ? (NQ/THREAD_W) : (NQ/THREAD_W+1);

    const dim3 block_conf{THREAD_W, THREAD_H};
    const dim3 grid_conf {BLOCK_W, 1, batch_size};
#endif

    AT_DISPATCH_FLOATING_TYPES(faces.scalar_type(), "forward_sd2mesh_cpu_kernel", [&]{
        using index_t = int64_t;

#ifdef ALGO_V1
        const int s_num_b   = block_conf.x*5 * sizeof(scalar_t);
        const int s_pad_b   = s_num_b % sizeof(int64_t);
        const int s_mem_b   = s_num_b + s_pad_b + block_conf.x*sizeof(int64_t);

        // int maxbytes = 98304; // 96 KB
        int maxbytes = s_mem_b; // 96 KB
        cudaFuncSetAttribute(sd2mesh_forward_cuda_impl<scalar_t, int64_t>, 
            cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
        sd2mesh_forward_cuda_impl<scalar_t, int64_t>
        <<<grid_conf, block_conf, s_mem_b>>>(
            faces.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            query.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),

            s_dist.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            tri_id.packed_accessor32<int64_t,  2, torch::RestrictPtrTraits>(),
            weight.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>()
        );
#elif ALGO_V2
        const int s_num_b   = block_conf.x * (3*3 + 7) * sizeof(scalar_t);

        // int maxbytes = 98304; // 96 KB
        int maxbytes = s_num_b; // 96 KB
        cudaFuncSetAttribute(sd2mesh_forward_cuda_impl_v2<scalar_t, int64_t>, 
            cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
        sd2mesh_forward_cuda_impl_v2<scalar_t, int64_t>
        <<<grid_conf, block_conf, s_num_b>>>(
            faces.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            query.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),

            s_dist.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            tri_id.packed_accessor32<int64_t,  2, torch::RestrictPtrTraits>(),
            weight.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>()
        );
#endif
    });

    return {s_dist, tri_id, weight};
}

// end of namespace
}