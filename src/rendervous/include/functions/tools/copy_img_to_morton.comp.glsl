#version 460
#extension GL_GOOGLE_include_directive : require
#include "../common_functions.h"

layout(binding = 1, std430) uniform Custom {
    float_ptr in_tensor; // img
    float_ptr out_tensor; // buffer
    int resolution; // power of 2
    int output_dim;
};

void main(){
    uvec3 thread_id;
    if (!start_function(thread_id))
    return;

    int index = int(thread_id.x);

    int px = unpart1by1(index);
    int py = unpart1by1(index >> 1);

    int offset_src = (py * resolution + px)*output_dim;
    int offset_dst = index * output_dim;

    [[unroll]]
    for (int i=0; i<output_dim; i++)
        out_tensor.data[offset_dst + i] = in_tensor.data[offset_src + i];
}