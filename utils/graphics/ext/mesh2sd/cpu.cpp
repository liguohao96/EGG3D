#include <torch/types.h>
#include <vector>
#include <cstdio>
#include <cmath>

// #define DEBUG

#include "utils.h"

namespace mesh2sd{

#define INFINITY 99999

template <typename scalar_t, typename index_t>
void sd2mesh_forward_cpu_impl(
    const torch::TensorAccessor<scalar_t, 4> faces, // b, nf, 3, 3
    const torch::TensorAccessor<scalar_t, 3> query, // b, n, 3

    torch::TensorAccessor<scalar_t,  2>      d_buf, // b, n
    torch::TensorAccessor<index_t,   2>      i_buf, // b, n
    torch::TensorAccessor<scalar_t,  3>      w_buf  // b, n, 3
){
    const int BATCH_SIZE = faces.size(0);
    const int NUM_FACES  = faces.size(1);
    const int NUM_QUERY  = query.size(1);

    // scalar_t EPS = 1e-5;

    for(int bi=0; bi<BATCH_SIZE; bi+=1){
        for(int qi=0; qi<NUM_QUERY; qi+=1){

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
#ifdef DEBUG
            printf("query xyz (%f, %f, %f)\n", xyz.x, xyz.y, xyz.z);
#endif

            for(int fi=0; fi<NUM_FACES; fi+=1){

                local_v33[0].x = faces[bi][fi][0][0];
                local_v33[0].y = faces[bi][fi][0][1];
                local_v33[0].z = faces[bi][fi][0][2];
                local_v33[1].x = faces[bi][fi][1][0];
                local_v33[1].y = faces[bi][fi][1][1];
                local_v33[1].z = faces[bi][fi][1][2];
                local_v33[2].x = faces[bi][fi][2][0];
                local_v33[2].y = faces[bi][fi][2][1];
                local_v33[2].z = faces[bi][fi][2][2];

                point_triangle_CP<scalar_t>(xyz, local_v33[0], local_v33[1], local_v33[2], out);
                
                scalar_t abs_d = fabs(out.w);

                bool is_closer = abs_d < u_dist - EPS;
                bool is_equal  = fabs(abs_d - u_dist) < EPS;
                bool is_signgt = sign(out.w) > sign(s_dist);   // new tirangle got + distance, origin is - distance

#ifdef DEBUG
                printf("u_dis/s_dis/abs() %f %f %f\n", u_dist, out.w, abs_d);
                printf("%d %d %d\n", is_closer, is_equal, is_signgt);
#endif

                bool change = is_closer || ( is_equal && is_signgt );

                if ( change ){
                    s_dist = out.w;
                    u_dist = abs_d;
                    w3.x   = out.x;
                    w3.y   = out.y;
                    w3.z   = out.x;
                    tid = fi;
                }
            }

            d_buf[bi][qi]    = s_dist;
            i_buf[bi][qi]    = tid;
            w_buf[bi][qi][0] = w3.x;
            w_buf[bi][qi][1] = w3.y;
            w_buf[bi][qi][2] = w3.z;
        }
    }
}

std::vector<torch::Tensor> sd_query_forward_cpu(
    torch::Tensor faces,
    torch::Tensor query
){
    // https://github.com/iamyoukou/sdf3d
    const unsigned int  BS         = faces.size(0);
    const unsigned int  num_tri    = faces.size(1);
    const unsigned int  NQ         = query.size(1);

    auto tri_id = torch::empty({BS, NQ},    faces.options().dtype(torch::kInt64));
    auto s_dist = torch::empty({BS, NQ},    faces.options());
    auto weight = torch::empty({BS, NQ, 3}, faces.options());

    AT_DISPATCH_FLOATING_TYPES(faces.scalar_type(), "forward_sd2mesh_cpu_kernel", [&]{

        sd2mesh_forward_cpu_impl<scalar_t, int64_t>(
            faces.accessor<scalar_t, 4>(),
            query.accessor<scalar_t, 3>(),

            s_dist.accessor<scalar_t, 2>(),
            tri_id.accessor<int64_t,  2>(),
            weight.accessor<scalar_t, 3>()
        );
    });

    return {s_dist, tri_id, weight};
}
// end of namespace
}