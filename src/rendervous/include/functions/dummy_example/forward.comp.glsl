#version 460
#extension GL_GOOGLE_include_directive : require
//#define LOCAL_SIZE_X 32
#include "../common_functions.h"
//#extension GL_NV_cooperative_matrix : enable
//#extension GL_KHR_memory_scope_semantics : enable
//#extension GL_EXT_shader_explicit_arithmetic_types : enable
//#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable
//#extension GL_KHR_shader_subgroup_basic : enable


layout(binding = 1) uniform Custom {
    float_ptr a;
    float_ptr b;
    float_ptr c;
    float scale;
};

//fcoopmatNV<32, gl_ScopeSubgroup, 16, 16> results[4];


void main(){
    uvec3 thread_id;
    if (!start_function(thread_id))
    return;

    c.data[thread_id.x] = b.data[thread_id.x] * scale + a.data[thread_id.x];

//    int group_id = int(gl_WorkGroupID.x);
//
//    for (int i=0; i<4; i++)
//    {
//        int block_start = group_id * 1024 + i * 256;
//
//        float_ptr a_buffer = float_ptr(a + block_start * 4);
//        float_ptr b_buffer = float_ptr(b + block_start * 4);
//
//        fcoopmatNV<32, gl_ScopeSubgroup, 16, 16> matA = fcoopmatNV<32, gl_ScopeSubgroup, 16, 16>(0.0);
//        coopMatLoadNV(matA, a_buffer.data, 0, 16, false);
//        fcoopmatNV<32, gl_ScopeSubgroup, 16, 16> matB = fcoopmatNV<32, gl_ScopeSubgroup, 16, 16>(0.0);
//        coopMatLoadNV(matB, b_buffer.data, 0, 16, false);
//        fcoopmatNV<32, gl_ScopeSubgroup, 16, 16> matC = matA + matB*scale;
//        results[i] = matC;
//    }
//
//    float_ptr c_buffer = float_ptr(c + group_id * 1024);
//    for (int i=0; i<4; i++)
//    {
//        //    fcoopmatNV<32, gl_ScopeSubgroup, 8, 8> matC = fcoopmatNV<32, gl_ScopeSubgroup, 8, 8>(3.14);
//        coopMatStoreNV(results[i], c_buffer.data, i*256, 16, false);
//        //c.data[thread_id] = scale * b.data[thread_id] + a.data[thread_id];
//    }
}