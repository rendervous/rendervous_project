#ifndef COMMON_H
#define COMMON_H

#include "supported.h"

// #define SUPPORTED_FLOAT_ATOM_ADD

#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_buffer_reference2: require
#extension GL_ARB_gpu_shader_int64 : require
#ifdef SUPPORTED_RAY_QUERY
#extension GL_EXT_ray_query : require
#extension GL_EXT_ray_tracing : require
#endif
#ifdef SUPPORTED_FLOAT_ATOM_ADD
#extension GL_EXT_shader_atomic_float : require
#endif
#extension GL_EXT_control_flow_attributes : require

#include "randoms.h"
#include "scattering_AD.h"

#define FORWARD void forward(map_object, float _input[INPUT_DIM], out float _output[OUTPUT_DIM])
#define BACKWARD void backward(map_object, float _input[INPUT_DIM], float _output_grad[OUTPUT_DIM], inout float _input_grad[INPUT_DIM])
#define BACKWARD_USING_OUTPUT void backward(map_object, float _input[INPUT_DIM], float _output[OUTPUT_DIM], float _output_grad[OUTPUT_DIM], inout float _input_grad[INPUT_DIM])

#define SAVE_SEED(v) uvec4 _rdv_saved_seed_##v = get_seed();
#define SET_SEED(v) set_seed(_rdv_saved_seed_##v);
#define BRANCH_SEED(v) uvec4 _rdv_saved_seed_##v = create_branch_seed();
#define SEED(v) _rdv_saved_seed_##v

#define BEGIN_BRANCH(v) uvec4 _rdv_saved_seed_##v = begin_branch_seed();
#define END_BRANCH(v) set_seed(_rdv_saved_seed_##v);


#define PRINT debugPrintfEXT
#ifdef ENABLE_ASSERT
#define ASSERT(cond, message) if (!(cond)) debugPrintfEXT(message);
#else
#define ASSERT(cond, message) ;
#endif

#define SWITCH_SEED(from_seed, to_seed) \
_rdv_saved_seed_##from_seed = get_seed(); \
set_seed(_rdv_saved_seed_##to_seed);

#define USING_SEED(secondary_seed, method_call) \
set_seed(_rdv_saved_seed_##secondary_seed); \
method_call;

#define USING_SECONDARY_SEED(main_seed, secondary_seed, method_call) \
_rdv_saved_seed_##main_seed = get_seed(); \
set_seed(_rdv_saved_seed_##secondary_seed); \
method_call; \
_rdv_saved_seed_##secondary_seed = get_seed(); \
set_seed(_rdv_saved_seed_##main_seed);

#define GPUPtr uint64_t

#define POINTER(type_name, align) layout(buffer_reference, scalar, buffer_reference_align=align) buffer type_name##_ptr { type_name data[]; };

POINTER(float, 4)
POINTER(int, 4)
POINTER(uint, 4)
POINTER(GPUPtr, 8)
POINTER(vec4, 4)
POINTER(vec3, 4)
POINTER(vec2, 4)
POINTER(mat2, 4)
POINTER(mat3, 4)
POINTER(mat4, 4)
POINTER(mat2x3, 4)
POINTER(mat2x4, 4)
POINTER(mat3x2, 4)
POINTER(mat3x4, 4)
POINTER(mat4x2, 4)
POINTER(mat4x3, 4)
POINTER(ivec4, 4)
POINTER(ivec3, 4)
POINTER(ivec2, 4)
POINTER(uvec4, 4)
POINTER(uvec3, 4)
POINTER(uvec2, 4)


// Predefined structs

struct Parameter
{
    GPUPtr data;
    uint stride[4];
    uint shape[4];
    GPUPtr grad_data;
};

layout(buffer_reference, scalar, buffer_reference_align=8) buffer MeshInfo {
    Parameter positions;
    Parameter normals;
    Parameter coordinates;
    Parameter tangents;
    Parameter binormals;
    Parameter indices;
};

layout(buffer_reference, scalar, buffer_reference_align=8) buffer RaycastableInfo {
    GPUPtr callable_map;
    GPUPtr mesh_info;
};

#define DECLARE_TENSOR(dim) \
float_ptr param_buffer(in Parameter tensor, int index[dim]) { \
    uint element_offset = 0; \
    [[unroll]] \
    for (int d = 0; d < dim; d++) element_offset += tensor.stride[d] * index[d]; \
    return float_ptr(tensor.data + (element_offset << 2)); \
}\
float_ptr param_grad_buffer(in Parameter tensor, int index[dim]) { \
    uint element_offset = 0; \
    [[unroll]] \
    for (int d = 0; d < dim; d++) element_offset += tensor.stride[d] * index[d]; \
    return float_ptr(tensor.grad_data + (element_offset << 2)); \
}

float param_float(in Parameter tensor)
{
    float_ptr buf = float_ptr(tensor.data);
    return buf.data[0];
}

vec2 param_vec2(in Parameter tensor)
{
    float_ptr buf = float_ptr(tensor.data);
    return vec2(buf.data[0], buf.data[1]);
}

vec3 param_vec3(in Parameter tensor)
{
    float_ptr buf = float_ptr(tensor.data);
    return vec3(buf.data[0], buf.data[1], buf.data[2]);
}

vec4 param_vec4(in Parameter tensor){
    float_ptr buf = float_ptr(tensor.data);
    return vec4(buf.data[0], buf.data[1], buf.data[2], buf.data[3]);
}

DECLARE_TENSOR(1)

DECLARE_TENSOR(2)

DECLARE_TENSOR(3)

DECLARE_TENSOR(4)

float_ptr param_buffer(in Parameter tensor, int index) {
    return param_buffer(tensor, int[1](index));
}
float_ptr param_buffer(in Parameter tensor, ivec2 index) {
    return param_buffer(tensor, int[2](index.y, index.x));
}
float_ptr param_buffer(in Parameter tensor, ivec3 index) {
    return param_buffer(tensor, int[3](index.z, index.y, index.x));
}
float_ptr param_buffer(in Parameter tensor, ivec4 index) {
    return param_buffer(tensor, int[4](index.w, index.z, index.y, index.x));
}

float_ptr param_grad_buffer(in Parameter tensor, int index) {
    return param_grad_buffer(tensor, int[1](index));
}
float_ptr param_grad_buffer(in Parameter tensor, ivec2 index) {
    return param_grad_buffer(tensor, int[2](index.y, index.x));
}
float_ptr param_grad_buffer(in Parameter tensor, ivec3 index) {
    return param_grad_buffer(tensor, int[3](index.z, index.y, index.x));
}
float_ptr param_grad_buffer(in Parameter tensor, ivec4 index) {
    return param_grad_buffer(tensor, int[4](index.w, index.z, index.y, index.x));
}

float atomicAdd_f(float_ptr buf, int index, float value)
{
    #ifdef SUPPORTED_FLOAT_ATOM_ADD
    return atomicAdd(buf.data[index], value);
    #else
    uint_ptr buf_as_uint = uint_ptr(uint64_t(buf));
    uint old = buf_as_uint.data[index];
    uint assumed;
    do {
        assumed = old;
        old = atomicCompSwap(buf_as_uint.data[index], assumed, floatBitsToUint(value + uintBitsToFloat(assumed)));
    } while(assumed != old);
    return uintBitsToFloat(old);
    #endif
}

#define IS_NULL(x) (GPUPtr(x) == 0)

#define REQUIRES_LINEAR(in_features, out_features) \
void linear_fw(map_object, in float x[(in_features)], out float y[(out_features)], GPUPtr weights, GPUPtr bias) { \
    float_ptr weights_buffer = float_ptr(weights); \
    float_ptr bias_buffer = float_ptr(bias); \
    int cw = 0; \
    int cb = 0; \
    [[unroll]] for (int i = 0; i < out_features; i++) { \
        y[i] = 0.0; \
        [[unroll]] for (int j=0; j<in_features; j++) y[i] += x[j] * weights_buffer.data[cw++]; \
        if (bias != 0) y[i] += bias_buffer.data[cb++]; \
    } \
} \
void linear_bw(map_object, in float x[in_features], in float de_dy[out_features], out float de_dx[in_features], GPUPtr weights, GPUPtr bias, GPUPtr de_dweights, GPUPtr de_dbias) { \
    [[unroll]] for (int j=0; j < in_features; j++) de_dx[j] = 0; \
    float_ptr weights_buffer = float_ptr(weights); \
    float_ptr dweight_buffer = float_ptr(de_dweights); \
    int N = out_features * in_features; \
    [[unroll]] for (int c = 0; c < N; c++) { \
        int i = c / in_features; \
        int j = c % in_features; \
        de_dx[j] += de_dy[i] * weights_buffer.data[c]; \
        atomicAdd_f(dweight_buffer, c, de_dy[i] * x[j]); \
    } \
    float_ptr dbias_buffer = float_ptr(de_dbias); \
    [[unroll]] for (int i=0; i<out_features; i++) atomicAdd_f(dbias_buffer, i, de_dy[i]); \
}


#define REQUIRES_MATMUL(in_features, out_features) \
void matmul_fw(map_object, in float x[(in_features)], out float y[(out_features)], GPUPtr weights) { \
    float_ptr weights_buffer = float_ptr(weights); \
    int cw = 0; \
    [[unroll]] for (int i = 0; i < out_features; i++) y[i] = 0.0;\
    [[unroll]] for (int j = 0; j < in_features; j++) \
    [[unroll]] for (int i = 0; i < out_features; i++) { \
         y[i] += x[j] * weights_buffer.data[cw++]; \
    } \
} \
void matmul_bw(map_object, in float x[in_features], in float de_dy[out_features], inout float de_dx[in_features], GPUPtr weights, GPUPtr de_dweights) { \
    float_ptr weights_buffer = float_ptr(weights); \
    float_ptr dweight_buffer = float_ptr(de_dweights); \
    int N = out_features * in_features; \
    [[unroll]] for (int c = 0; c < N; c++) { \
        int j = c / out_features; \
        int i = c % out_features; \
        de_dx[j] += de_dy[i] * weights_buffer.data[c]; \
        atomicAdd_f(dweight_buffer, c, de_dy[i] * x[j]); \
    } \
}\
void pre_matmul_fw(map_object, in float x[(in_features)], out float y[(out_features)], GPUPtr weights) { \
    float_ptr weights_buffer = float_ptr(weights); \
    int cw = 0; \
    [[unroll]] for (int i = 0; i < out_features; i++) { \
    y[i] = 0.0;\
    [[unroll]] for (int j = 0; j < in_features; j++) \
         y[i] += x[j] * weights_buffer.data[cw++]; \
    } \
} \
void pre_matmul_bw(map_object, in float x[in_features], in float de_dy[out_features], inout float de_dx[in_features], GPUPtr weights, GPUPtr de_dweights) { \
    float_ptr weights_buffer = float_ptr(weights); \
    float_ptr dweight_buffer = float_ptr(de_dweights); \
    int N = out_features * in_features; \
    [[unroll]] for (int c = 0; c < N; c++) { \
        int i = c / in_features; \
        int j = c % in_features; \
        de_dx[j] += de_dy[i] * weights_buffer.data[c]; \
        atomicAdd_f(dweight_buffer, c, de_dy[i] * x[j]); \
    } \
}


#define ACTIVATION_FUNCTION \
float activation_fw(map_object, float x); \
void activation_bw(map_object, float x, float dL_dy, inout float dL_dx); \
FORWARD { [[unroll]] for (int i=0; i<INPUT_DIM; i++) _output[i] = activation_fw(object, _input[i]); } \
BACKWARD { [[unroll]] for (int i=0; i<INPUT_DIM; i++) activation_bw(object, _input[i], _output_grad[i], _input_grad[i]); }


struct Surfel
{
    vec3 P; // Position at the surface
    vec3 N; // Shading normal at the surface (might differ from real normal G)
    vec3 G; // Gradient vector at the surface pointing 'outside'
    vec2 C; // Coordinates to parameterize the surface
    vec3 T; // Tangent vector of the parameterization
    vec3 B; // Binormal vector of the parameterization
};

//struct HitInfo
//{
//    float t;
//    vec3 N;
//    vec3 T;
//    vec3 B;
//    vec2 C;
//    int patch_index;
//};

bool hit2surfel (vec3 x, vec3 w, in float[16] a, out float t, out int patch_index, out Surfel surfel)
{
    patch_index = floatBitsToInt(a[15]);
    if (patch_index == -1)
        return false;
    t = a[0];
    surfel.P = w * t + x;
    surfel.N = vec3(a[1], a[2], a[3]);
    surfel.G = vec3(a[4], a[5], a[6]);
    surfel.C = vec2(a[7], a[8]);
    surfel.T = vec3(a[9], a[10], a[11]);
    surfel.B = vec3(a[12], a[13], a[14]);
    return true;
}

void surfel2array(float t, int patch_index, Surfel surfel, out float[16] a)
{
    a = float[16](
        t,
        surfel.N.x,surfel.N.y,surfel.N.z,
        surfel.G.x,surfel.G.y,surfel.G.z,
        surfel.C.x,surfel.C.y,
        surfel.T.x,surfel.T.y,surfel.T.z,
        surfel.B.x,surfel.B.y,surfel.B.z,
        intBitsToFloat(patch_index)
    );
}

void noHit2array(out float[16] a)
{
    a = float[16](
        0,
        0, 0, 0,
        0, 0, 0,
        0, 0,
        0, 0, 0,
        0, 0, 0,
        intBitsToFloat(-1)
    );
}

//void array2hit (in float[13] a, out HitInfo info)
//{
//    info.t = a[0];
//    info.N = vec3(a[1], a[2], a[3]);
//    info.T = vec3(a[4], a[5], a[6]);
//    info.B = vec3(a[7], a[8], a[9]);
//    info.C = vec2(a[10], a[11]);
//    info.patch_index = floatBitsToInt(a[12]);
//}

//void hit2array(in HitInfo info, out float[13] a)
//{
//    a = float[13](
//    info.t,
//    info.N.x, info.N.y, info.N.z,
//    info.T.x, info.T.y, info.T.z,
//    info.B.x, info.B.y, info.B.z,
//    info.C.x, info.C.y,
//    intBitsToFloat(info.patch_index)
//    );
//}

//void hit2surfel(vec3 x, vec3 w, in HitInfo info, out SurfelData surfel)
//{
//    surfel.P = w * info.t + x;
//    surfel.N = info.N;
//    surfel.T = info.T;
//    surfel.B = info.B;
//    surfel.C = info.C;
//}

vec3 transform_position(vec3 P, mat4 T)
{
    return (T * vec4(P, 1)).xyz;
}

vec3 transform_position(vec3 P, mat4x3 T)
{
    return (T * vec4(P, 1));
}

vec3 transform_normal(vec3 N, mat4 T)
{
    mat3 T_N = mat3(T);
    return normalize(transpose(inverse(T_N)) * N);
}

vec3 transform_normal(vec3 N, mat4x3 T)
{
    mat3 T_N = mat3(T);
    return normalize(transpose(inverse(T_N)) * N);
}

vec3 transform_direction(vec3 D, mat4 T)
{
    return (T * vec4(D, 0)).xyz;
}

vec3 transform_direction(vec3 D, mat4x3 T)
{
    return (T * vec4(D, 0));
}

Surfel transform(Surfel surfel, mat4 T)
{
    Surfel result;
    result.P = transform_position(surfel.P, T);
    result.N = transform_normal(surfel.N, T);
    result.G = transform_normal(surfel.G, T);
    result.C = surfel.C;
    result.T = transform_direction(surfel.T, T);
    result.B = transform_direction(surfel.B, T);
    return result;
}

Surfel transform(Surfel surfel, mat4x3 T)
{
    Surfel result;
    result.P = transform_position(surfel.P, T);
    result.N = transform_normal(surfel.N, T);
    result.G = transform_normal(surfel.G, T);
    result.C = surfel.C;
    result.T = transform_direction(surfel.T, T);
    result.B = transform_direction(surfel.B, T);
    return result;
}

//HitInfo transform_hit(HitInfo info, mat4 T)
//{
//    HitInfo result;
//    result.t = info.t;
//    result.N = transform_normal(info.N, T);
//    result.T = transform_direction(info.T, T);
//    result.B = transform_direction(info.B, T);
//    result.C = info.C;
//    result.patch_index = info.patch_index;
//    return result;
//}
//
//HitInfo transform_hit(HitInfo info, mat4x3 T)
//{
//    HitInfo result;
//    result.t = info.t;
//    result.N = transform_normal(info.N, T);
//    result.T = transform_direction(info.T, T);
//    result.B = transform_direction(info.B, T);
//    result.C = info.C;
//    result.patch_index = info.patch_index;
//    return result;
//}

mat4x3 inverse_transform(mat4x3 T)
{
    mat4 M = mat4(T);
    M[3][3] = 1.0;
    M = inverse(M);
    return mat4x3(M);
}

//HitInfo create_no_hit()
//{
//    return HitInfo(0.0, vec3(0.0), vec3(0.0), vec3(0.0), vec2(0.0), -1);
//}

bool null_scatter(vec3 wi, vec3 wo)
{
    return wi == wo;
}

Surfel sample_surfel(in MeshInfo mesh, int index, vec2 baricentrics)
{
    vec3 alphas = vec3(1 - baricentrics.x - baricentrics.y, baricentrics.x, baricentrics.y);

    int idx0, idx1, idx2;

    int i = index * 3;
    if (mesh.indices.data != 0)
    {
        int_ptr idxs = int_ptr(mesh.indices.data);
        idx0 = idxs.data[i++];
        idx1 = idxs.data[i++];
        idx2 = idxs.data[i];
    }
    else
    {
        idx0 = i++;
        idx1 = i++;
        idx2 = i;
    }

    vec3_ptr pos = vec3_ptr(mesh.positions.data);
    vec3 P = pos.data[idx0] * alphas.x + pos.data[idx1] * alphas.y + pos.data[idx2] * alphas.z;
    vec3 Nface = normalize(cross(pos.data[idx1] - pos.data[idx0], pos.data[idx2] - pos.data[idx0]));
    vec3_ptr nor = vec3_ptr(mesh.normals.data);
    vec3 N = mesh.normals.data == 0 ? Nface : normalize(nor.data[idx0] * alphas.x + nor.data[idx1] * alphas.y + nor.data[idx2] * alphas.z);
    vec2_ptr coordinates = vec2_ptr(mesh.coordinates.data);
    vec2 C = mesh.coordinates.data == 0 ? vec2(0.0) : coordinates.data[idx0] * alphas.x + coordinates.data[idx1] * alphas.y + coordinates.data[idx2] * alphas.z;
    vec3_ptr tang = vec3_ptr(mesh.tangents.data);
    vec3 T = mesh.tangents.data == 0 ? vec3(0.0) : tang.data[idx0] * alphas.x + tang.data[idx1] * alphas.y + tang.data[idx2] * alphas.z;
    vec3_ptr bin = vec3_ptr(mesh.binormals.data);
    vec3 B = mesh.binormals.data == 0 ? vec3(0.0) : bin.data[idx0] * alphas.x + bin.data[idx1] * alphas.y + bin.data[idx2] * alphas.z;

    return Surfel(P, N, Nface, C, T, B);
}

void sample_surfel_bw(in MeshInfo mesh, int primitive_index, vec2 baricentrics, in Surfel surfel_grad)
{
    // TODO: backprop surfel grads to mesh parameters
}


#define INCLUDE_ADDITIONAL_COLLISION_VARS bool from_outside = dot(win, surfel.G) < 0;\
    bool correct_hemisphere = (dot(win, surfel.N) < 0) == from_outside;\
    vec3 N = correct_hemisphere ? surfel.N : surfel.G;\
    vec3 fN = from_outside ? N : -N;



#endif