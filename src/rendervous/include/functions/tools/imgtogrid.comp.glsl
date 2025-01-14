#version 460
#extension GL_GOOGLE_include_directive : require
#include "../common_functions.h"

layout(binding = 1, std430) uniform Custom {
    float_ptr in_tensor; // img
    float_ptr out_tensor; // grid
    int[4] shape; // shape of img
    int dim;
    int output_dim;
};

int img_index(ivec2 c) {
    c = clamp(c, ivec2(0,0), ivec2(shape[1] - 1, shape[0] - 1));
    return c.x + c.y * shape[1];
}

int img_index(ivec3 c) {
    c = clamp(c, ivec3(0,0,0), ivec3(shape[2] - 1, shape[1] - 1, shape[0] - 1));
    return c.x + c.y * shape[2] + c.z * shape[1]*shape[2];
}

void main() {
    uvec3 thread_id;
    if (!start_function(thread_id))
    return;

    int index = int(thread_id.x);

    switch (dim)
    {
        case 1:
        int index0 = max(0, index - 1);
        int index1 = min(index, shape[0] - 1);
        [[unroll]]
        for (int i=0; i<output_dim; i++)
            out_tensor.data[index * output_dim + i] = (in_tensor.data[index0 * output_dim + i] + in_tensor.data[index1 * output_dim + i]) * 0.5;
        break;
        case 2:
        ivec2 c = ivec2(index % (shape[1] + 1), index / (shape[1] + 1));
        int index00 = img_index(c);
        int index01 = img_index(c - ivec2(1, 0));
        int index10 = img_index(c - ivec2(0, 1));
        int index11 = img_index(c - ivec2(1, 1));
        [[unroll]]
        for (int i=0; i<output_dim; i++)
            out_tensor.data[index * output_dim + i] = (
            in_tensor.data[index00 * output_dim + i] +
            in_tensor.data[index01 * output_dim + i] +
            in_tensor.data[index10 * output_dim + i] +
            in_tensor.data[index11 * output_dim + i]) * 0.25;
        break;
        case 3:
        ivec3 v = ivec3(index % (shape[2]+1), index / (shape[2]+1) % (shape[1]+1), index / ((shape[1]+1) * (shape[2]+1)));
        int index000 = img_index(v);
        int index001 = img_index(v - ivec3(1, 0, 0));
        int index010 = img_index(v - ivec3(0, 1, 0));
        int index011 = img_index(v - ivec3(1, 1, 0));
        int index100 = img_index(v - ivec3(0, 0, 1));
        int index101 = img_index(v - ivec3(1, 0, 1));
        int index110 = img_index(v - ivec3(0, 1, 1));
        int index111 = img_index(v - ivec3(1, 1, 1));
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