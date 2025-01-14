#version 460
#extension GL_GOOGLE_include_directive : require
#include "../common_functions.h"

layout(binding = 1, std430) uniform Custom {
    float_ptr in_tensor; // grid tensor
    float_ptr out_tensor; // img tensor
    int[4] shape; // grid shape
    int dim;
    int output_dim;
};

int index_imgtogrid_2d(int index){
    ivec2 c = ivec2(index % (shape[1] - 1), index / (shape[1] - 1));
    return c.y * shape[1] + c.x;
}

int index_imgtogrid_3d(int index){
    ivec3 c = ivec3(index % (shape[2] - 1), index / (shape[2] - 1) % (shape[1] - 1), index / ((shape[2]-1)*(shape[1]-1)));
    return c.z * shape[2] * shape[1] + c.y * shape[2] + c.x;
}


void main() {
    uvec3 thread_id;
    if (!start_function(thread_id))
    return;

    int index = int(thread_id.x);

    switch (dim)
    {
        case 1:
        int index0 = index;
        int index1 = index + 1;
        [[unroll]]
        for (int i=0; i<output_dim; i++)
            out_tensor.data[index * output_dim + i] = (in_tensor.data[index0 * output_dim + i] + in_tensor.data[index1 * output_dim + i]) * 0.5;
        break;
        case 2:
        int index00 = index_imgtogrid_2d(index);
        int index01 = index00 + 1;
        int index10 = index00 + shape[1];
        int index11 = index00 + shape[1] + 1;
        [[unroll]]
        for (int i=0; i<output_dim; i++)
            out_tensor.data[index * output_dim + i] = (
            in_tensor.data[index00 * output_dim + i] +
            in_tensor.data[index01 * output_dim + i] +
            in_tensor.data[index10 * output_dim + i] +
            in_tensor.data[index11 * output_dim + i]) * 0.25;
        break;
        case 3:
        int slice_stride = shape[1] * shape[2];
        int index000 = index_imgtogrid_3d(index);;
        int index001 = index000 + 1;
        int index010 = index000 + shape[2];
        int index011 = index000 + shape[2] + 1;
        int index100 = index000 + slice_stride;
        int index101 = index000 + 1 + slice_stride;
        int index110 = index000 + shape[2] + slice_stride;
        int index111 = index000 + shape[2] + 1 + slice_stride;
        [[unroll]]
        for (int i=0; i<output_dim; i++)
            out_tensor.data[index * output_dim + i] = (
            in_tensor.data[index000 * output_dim + i] +
            in_tensor.data[index001 * output_dim + i] +
            in_tensor.data[index010 * output_dim + i] +
            in_tensor.data[index011 * output_dim + i] +
            in_tensor.data[index100 * output_dim + i] +
            in_tensor.data[index101 * output_dim + i] +
            in_tensor.data[index110 * output_dim + i] +
            in_tensor.data[index111 * output_dim + i]
            ) * 0.125;
        break;
    }
}