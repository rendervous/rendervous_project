#version 460
#extension GL_GOOGLE_include_directive : require
#include "../common_functions.h"

layout(binding = 1, std430) uniform Custom {
    int_ptr out_tensor;
    int[4] shape;
    int dim;
};

void main(){
    uvec3 thread_id;
    if (!start_function(thread_id))
    return;

    int index = int(thread_id.x);
    [[unroll]]
    for (int i=0; i<dim; i++)
    {
        out_tensor.data[(index * dim + i)*2] = int (random() * shape[i]);
        out_tensor.data[(index * dim + i)*2+1] = 0;
    }
}