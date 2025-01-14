#version 460
#extension GL_GOOGLE_include_directive : require
#include "../common_functions.h"

layout(binding = 1, std430) uniform Custom {
    float_ptr in_tensor; // coordinates
    float_ptr out_tensor; // directions
};

void main() {
    uvec3 thread_id;
    if (!start_function(thread_id))
    return;

    int index = int(thread_id.x);

    vec2 c = vec2(in_tensor.data[index*2], in_tensor.data[index*2 + 1]);
    vec3 w = oct2dir(c);
    out_tensor.data[index*3 + 0] = w.x;
    out_tensor.data[index*3 + 1] = w.y;
    out_tensor.data[index*3 + 2] = w.z;
}