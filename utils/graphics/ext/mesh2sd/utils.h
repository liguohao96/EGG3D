#ifndef __UTILS_H__
#define __UTILS_H__

#ifdef __CUDACC__ // CUDA
    #define DEVICE __device__ __host__
    #define sign(x) copysignf(1.0f, x)

#else  // CPU
    #define DEVICE 
    using std::max;
    using std::min;
    #define sign(x) std::copysign(1.0, x)
#endif // CUDACC

#define min3(a,b,c) (min(min(a,b), c))
#define max3(a,b,c) (max(max(a,b), c))
#define EPS (1e-5)

template<typename T>
struct vec2 {
    T x;
    T y;
};

template<typename T>
DEVICE inline T vec2_cross(const vec2<T>& lhs, const vec2<T>& rhs) {
    return (lhs.x*rhs.y - rhs.x*lhs.y);
};

template<typename T>
DEVICE inline vec2<T> make_vec2(const T& x, const T& y){
    vec2<T> v2;
    v2.x = x;
    v2.y = y;
    return v2;
};

template<typename T>
struct vec3 {
    T x, y, z;
    // inline vec3(const T x, const T y, const T z):x(x), y(y), z(z){};
    // inline vec3():vec3{0, 0, 0}{};

    DEVICE inline T norm() const {
        T ret;

        // ret = sqrt(dot(*this));
#ifdef __CUDACC__ // CUDA
        ret = norm3df(x, y, z);
#else
        ret = sqrt(dot(*this));
#endif
        return ret;
    };
    DEVICE inline T rnorm() const {
        T ret;
        // ret = 1/norm();

#ifdef __CUDACC__ // CUDA
        ret = rnorm3df(x, y, z);
#else
        ret = 1/norm();
#endif
        return ret;
    };

    DEVICE inline vec3<T> cross(const vec3<T>& rhs) const {
        vec3<T> v3;
        v3.x =   y*rhs.z - z*rhs.y;
        // v3.y = -(x*rhs.z - z*rhs.x);
        v3.y =   z*rhs.x - x*rhs.z;
        v3.z =   x*rhs.y - y*rhs.x;
        return v3;
    };
    DEVICE inline T dot(const vec3<T>& rhs) const{
        T ret = static_cast<T>(0);
        // ret += x * rhs.x;
        // ret += y * rhs.y;
        // ret += z * rhs.z;

#ifdef __CUDACC__ // CUDA
        // ret = x * rhs.x;
        ret = fmaf(x, rhs.x, ret);
        ret = fmaf(y, rhs.y, ret);
        ret = fmaf(z, rhs.z, ret);
#else
        ret += x * rhs.x;
        ret += y * rhs.y;
        ret += z * rhs.z;
#endif
        return ret;
    };
    DEVICE inline vec3<T> operator-(const vec3<T>& rhs) const{
        // const auto lhs = *this;
        vec3<T> v3;
        v3.x = x - rhs.x;
        v3.y = y - rhs.y;
        v3.z = z - rhs.z;
        return v3;
    };
    DEVICE inline vec3<T> operator+(const vec3<T>& rhs) const{
        vec3<T> v3;
        v3.x = x + rhs.x;
        v3.y = y + rhs.y;
        v3.z = z + rhs.z;
        return v3;
    };
    DEVICE inline vec3<T> operator*(const T rhs) const{
        // const auto lhs = *this;
        vec3<T> v3;
        v3.x = x * rhs;
        v3.y = y * rhs;
        v3.z = z * rhs;
        return v3;
    };
};
template<typename T>
DEVICE inline vec3<T> make_vec3(const T& x, const T& y, const T& z){
    vec3<T> v3;
    v3.x = x;
    v3.y = y;
    v3.z = z;
    return v3;
};
template<typename T>
DEVICE inline vec3<T> vec3_cross(const vec3<T>& lhs, const vec3<T>& rhs) {
    vec3<T> v3;
    v3.x =   lhs.y*rhs.z - lhs.z*rhs.y;
    v3.y = -(lhs.x*rhs.z - lhs.z*rhs.x);
    v3.z =   lhs.x*rhs.y - lhs.y*rhs.x;
    return v3;
};
template<typename T>
DEVICE inline T vec3_dot(const vec3<T>& lhs, const vec3<T>& rhs) {
    T ret = static_cast<T>(0);
    ret += lhs.x * rhs.x;
    ret += lhs.y * rhs.y;
    ret += lhs.z * rhs.z;
    return ret;
};
template<typename T>
DEVICE inline vec3<T> vec3_sub(const vec3<T>& lhs, const vec3<T>& rhs) {
    // const auto lhs = *this;
    vec3<T> v3;
    v3.x = lhs.x - rhs.x;
    v3.y = lhs.y - rhs.y;
    v3.z = lhs.z - rhs.z;
    return v3;
};
template<typename T>
DEVICE inline vec3<T> vec3_add(const vec3<T>& lhs, const vec3<T>& rhs) {
    vec3<T> v3;
    v3.x = lhs.x + rhs.x;
    v3.y = lhs.y + rhs.y;
    v3.z = lhs.z + rhs.z;
    return v3;
};
template<typename T>
DEVICE inline vec3<T> vec3_mul(const vec3<T>& lhs, T rhs) {
    // const auto lhs = *this;
    vec3<T> v3;
    v3.x = lhs.x * rhs;
    v3.y = lhs.y * rhs;
    v3.z = lhs.z * rhs;
    return v3;
};

template<typename T>
struct vec4 {
    T x;
    T y;
    T z;
    T w;
    DEVICE inline vec3<T> xyz() const{
        vec3<T> v3;
        v3.x = x;
        v3.y = y;
        v3.z = z;
        return v3;
    };
    // DEVICE inline vec4<T> operator*(const T rhs) const{
    //     // const auto lhs = *this;
    //     vec4<T> v4;
    //     v4.x = x * rhs;
    //     v4.y = y * rhs;
    //     v4.z = z * rhs;
    //     v4.w = w * rhs;
    //     return v4;
    // };
};

template<typename T>
struct vec7 {
    T x, y, z, w, a, b, c;
};

// https://box2d.org/files/ErinCatto_GJK_GDC2010.pdf

template <typename scalar_t>
DEVICE __inline__ void point_line_CP(
    const vec3<scalar_t>& xyz,
    const vec3<scalar_t>& p0,
    const vec3<scalar_t>& p1,
    vec3<scalar_t>& out 
){
    // Flops ~32
    vec3<scalar_t> n  = p1 - p0;             // add 3
    vec3<scalar_t> v0 = xyz - p0;            // add 3
    // vec3<scalar_t> v1 = xyz - p1;
    // scalar_t denom = 1 / ( sqrt(n.dot(n)) + EPS); // n.length()*n.length() mul 3/add 3/sqrt 1/inv1
    // scalar_t denom = 1 / ( n.dot(n) ); // n.length()*n.length() mul 3/add 3/sqrt 1/inv1
    scalar_t denom = powf(n.rnorm(), 2); // n.length()*n.length() mul 3/add 3/sqrt 1/inv1
    scalar_t u, v;

    v = (v0).dot(n) * denom;                      // mul 3/add 2/mul 1
    u = 1 - v;                                    // add 1

    scalar_t p_u, p_v;
    p_u = fminf(fmaxf(u, static_cast<scalar_t>(0) ), static_cast<scalar_t>(1));
    p_v = 1 - p_u;                                // add 1

    // xyz - (u*p0 + v*p1) => xyz - (u*p0 + v*p1 + v*p0 - v*p0) => (xyz-p0) - v*(p1-p0) => (xyz-p1) + u*(p1-p0)
    vec3<scalar_t> diff = v0 - n*p_v;             // mul 3/add 3

#ifdef DEBUG 
    vec3<scalar_t> proj = p0*p_u + p1*p_v;     
    vec3<scalar_t> np_v = n*p_v;               
    printf("xyz   (%f, %f, %f)\n", xyz.x, xyz.y, xyz.z);
    printf("p0   (%f, %f, %f)\n", p0.x, p0.y, p0.z);
    printf("v0   (%f, %f, %f)\n", v0.x, v0.y, v0.z);
    printf("np_v (%f, %f, %f)\n", np_v.x, np_v.y, np_v.z);
    printf("diff (%f, %f, %f)\n", diff.x, diff.y, diff.z);
    printf("proj (%f, %f, %f)\n", proj.x, proj.y, proj.z);
#endif

    out.x = u;
    out.y = v;
    out.z = diff.norm();                 // mul 3/add 2/sqrt 1
    // out.z = sqrt(diff.dot(diff));                 // mul 3/add 2/sqrt 1
};

template <typename scalar_t>
DEVICE __inline__ void point_triangle_CP(
    const vec3<scalar_t>& xyz,
    const vec3<scalar_t>& v0,
    const vec3<scalar_t>& v1,
    const vec3<scalar_t>& v2,
#ifdef __CUDACC__
    const vec7<scalar_t>& pre,
#endif
    vec4<scalar_t>& out
){
    // Flops ~200(32*3+102)

    vec3<scalar_t> v0_v1  = v1 - v0; // add 3
    vec3<scalar_t> v0_v2  = v2 - v0; // add 3
    vec3<scalar_t> v0_xyz = xyz - v0; // add 3

    vec3<scalar_t> o01, o12, o20;

    point_line_CP(xyz, v0, v1, o01);
    point_line_CP(xyz, v1, v2, o12);
    point_line_CP(xyz, v2, v0, o20);

    scalar_t dist_list[5];
    int ri = 0;

    scalar_t u, v, w;

#ifdef __CUDACC__

    /* may be fast, but wrong result? */
    // vec3<scalar_t> normal;
    // normal.x = pre.x;
    // normal.y = pre.y;
    // normal.z = pre.z;
    // scalar_t invDet   = pre.w;
    // scalar_t       dist = v0_xyz.dot(normal);         // mul 3/add 2

    // // https://math.stackexchange.com/questions/544946/determine-if-projection-of-3d-point-onto-plane-is-within-a-triangle
    // // w = normal.dot( (v0_v1).cross(v0_xyz) ) * invDet;
    // // v = normal.dot( (v0_xyz).cross(v0_v2) ) * invDet;
    // // u = 1 - v - w;                                                 // add
    // // printf("old uvw (%f, %f, %f)\n", u, v, w);

    // // https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
    // // scalar_t d00 = v0_v1.dot(v0_v1);
    // // scalar_t d01 = v0_v1.dot(v0_v2);
    // // scalar_t d11 = v0_v2.dot(v0_v2);
    // scalar_t d00 = pre.a;
    // scalar_t d01 = pre.b;
    // scalar_t d11 = pre.c;
    // scalar_t d20 = v0_xyz.dot(v0_v1);
    // scalar_t d21 = v0_xyz.dot(v0_v2);
    // // scalar_t denom = d00 * d11 - d01 * d01;
    // scalar_t invDenom = pre.w;
    // v = (d11 * d20 - d01 * d21) * invDenom;
    // w = (d00 * d21 - d01 * d20) * invDenom;
    // u = 1 - v - w;

    /* may be slow, but the same with CPU */
    vec3<scalar_t> normal = v0_v1.cross(v0_v2);       // mul 6/add 3
    scalar_t invDet       = normal.rnorm();

    normal = normal * invDet;                         // mul 3

    scalar_t       dist = v0_xyz.dot(normal);         // mul 3/add 2

    vec3<scalar_t> proj = xyz - normal*dist;          // mul 3/add 3

    // vec3<scalar_t> p01_3 = (v0_v1).cross(proj - v0);     // mul 6/add 3
    vec3<scalar_t> p20_3 = (proj - v0).cross(v0_v2);     // mul 6/add 3
    vec3<scalar_t> p12_3 = (v1 - proj).cross(v2 - proj); // mul 6/add 3

    // u = sign(p12_3.dot(normal)) * sqrt(p12_3.dot(p12_3)) * invDet; // sign/mul 3/add 2/mul/sqrt/mul 3/add 2/mul
    // v = sign(p20_3.dot(normal)) * sqrt(p20_3.dot(p20_3)) * invDet; // sign/mul 3/add 2/mul/sqrt/mul 3/add 2/mul
    u = sign(p12_3.dot(normal)) * p12_3.norm() * invDet; // sign/mul 3/add 2/mul/sqrt/mul 3/add 2/mul
    v = sign(p20_3.dot(normal)) * p20_3.norm() * invDet; // sign/mul 3/add 2/mul/sqrt/mul 3/add 2/mul
    w = 1 - u - v;                                                 // add
#else
    vec3<scalar_t> normal = v0_v1.cross(v0_v2);       // mul 6/add 3
    scalar_t invDet       = normal.rnorm();

    normal = normal * invDet;                         // mul 3

    scalar_t       dist = v0_xyz.dot(normal);         // mul 3/add 2

    vec3<scalar_t> proj = xyz - normal*dist;          // mul 3/add 3

    // vec3<scalar_t> p01_3 = (v0_v1).cross(proj - v0);     // mul 6/add 3
    vec3<scalar_t> p20_3 = (proj - v0).cross(v0_v2);     // mul 6/add 3
    vec3<scalar_t> p12_3 = (v1 - proj).cross(v2 - proj); // mul 6/add 3

    // u = sign(p12_3.dot(normal)) * sqrt(p12_3.dot(p12_3)) * invDet; // sign/mul 3/add 2/mul/sqrt/mul 3/add 2/mul
    // v = sign(p20_3.dot(normal)) * sqrt(p20_3.dot(p20_3)) * invDet; // sign/mul 3/add 2/mul/sqrt/mul 3/add 2/mul
    u = sign(p12_3.dot(normal)) * p12_3.norm() * invDet; // sign/mul 3/add 2/mul/sqrt/mul 3/add 2/mul
    v = sign(p20_3.dot(normal)) * p20_3.norm() * invDet; // sign/mul 3/add 2/mul/sqrt/mul 3/add 2/mul
    w = 1 - u - v;                                                 // add
#endif

    // dist_list[0] = fabsf(dist);                           // mul 3/add 2
    // dist_list[1] = o01.z;
    // dist_list[2] = o12.z;
    // dist_list[4] = o20.z;
    dist_list[0] = dist;                           // mul 3/add 2
    dist_list[1] = o01.z*sign(dist);
    dist_list[2] = o12.z*sign(dist);
    dist_list[4] = o20.z*sign(dist);

    int i0=0,i1=0,i2=0,i4=0;

// #ifdef __CUDACC__
//     i1 = w < 0;
//     i2 = (w >= 0 && u < 0);
//     i4 = (w >= 0 && u >= 0 && v < 0);

//     // i1 = (o01.y < 0 && o20.x < 0);
//     // i2 = (o01.x < 0 && o12.y < 0);
//     // i4 = (o12.x < 0 && o20.y < 0);

//     // i1 += (o01.x >= 0 && o01.y >= 0 && w <= 0);
//     // i2 += (o12.x >= 0 && o12.y >= 0 && u <= 0);
//     // i4 += (o20.x >= 0 && o20.y >= 0 && v <= 0);

//     // i0 = (u > 0 && v > 0 && w > 0);

//     ri = (i1) + (i2<<1) + (i4<<2);
// #else
    i1 = (o01.y < 0 && o20.x < 0);
    i2 = (o01.x < 0 && o12.y < 0);
    i4 = (o12.x < 0 && o20.y < 0);

    i1 += (o01.x >= 0 && o01.y >= 0 && w <= 0);
    i2 += (o12.x >= 0 && o12.y >= 0 && u <= 0);
    i4 += (o20.x >= 0 && o20.y >= 0 && v <= 0);

    // i0 = (u > 0 && v > 0 && w > 0);

    ri = (~i0) & (i1 | i2<<1 | i4<<2);
// #endif


#ifdef DEBUG 
    printf("AB (%f, %f, %f)\n", o01.x, o01.y, o01.z);
    printf("BC (%f, %f, %f)\n", o12.x, o12.y, o12.z);
    printf("CA (%f, %f, %f)\n", o20.x, o20.y, o20.z);
    printf("projected   (%f, %f, %f)\n", proj.x, proj.y, proj.z);
    printf("barycentric (%f, %f, %f)\n", u, v, w);
    printf("ri=%d dist=", ri);
    for(int i=0;i<7;++i)
        printf("%f,", dist_list[i]);
    printf("\n");
#endif

    out.x = u;
    out.y = v;
    out.z = w;
    // out.w = sign(dist)*dist_list[ri];
    out.w = dist_list[ri];
};

#endif