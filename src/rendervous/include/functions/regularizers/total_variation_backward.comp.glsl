#version 460
#extension GL_GOOGLE_include_directive : require
#include "../common_functions.h"

layout(binding = 1, std430) uniform Custom {
    GPUPtr in_tensor; // image or volume to compute the TV
    GPUPtr out_grad_tensor; // same dimensions of the input but 1 channel
    GPUPtr in_grad_tensor;
    int[4] shape; // shape of the input
    int dim; // number of coordinates of input. e.g., 2 for images, 3 for volumes
};


bool adjacent_index(int index, int dc[4], out int adj_index)
{
    int size = 1;
    adj_index = 0;
    [[unroll]]
    for (int i=0; i<dim; i++)
    {
        int id = shape[dim-i-1];
        int pc = index % id; // current coordinate
        pc += dc[i];
        if (pc < 0 || pc >= id)
        return false;

        adj_index += pc * size;
        index /= id;
        size *= id;
    }
    return true;
}

void total_variation_bw(int index, int adj_index, float dL_dtv)
{
    int channels = shape[dim];
    float_ptr center_value = float_ptr(in_tensor + index * channels * 4);
    float_ptr adj_value = float_ptr(in_tensor + adj_index * channels * 4);
    float_ptr center_grad = float_ptr(in_grad_tensor + index * channels * 4);
    float_ptr out_adj_grad_buf = float_ptr(out_grad_tensor + adj_index * 4);
    [[unroll]]
    for (int i=0; i<channels; i++)
        center_grad.data[i] += (dL_dtv + out_adj_grad_buf.data[0]) * sign(center_value.data[i] - adj_value.data[i]);
}

void main() {
    uvec3 thread_id;
    if (!start_function(thread_id))
    return;

    int index = int(thread_id.x);

    float_ptr out_grad_buf = float_ptr(out_grad_tensor + index * 4);
    float dL_dtv = out_grad_buf.data[0];

    float_ptr in_grad_buf = float_ptr(in_grad_tensor + shape[dim] * index * 4);
    [[unroll]]
    for (int i=0; i<shape[dim]; i++)
        in_grad_buf.data[i] = 0.0; // initialize grads

    int dc[4] = {0, 0, 0, 0};
    int adj_index;
    [[unroll]]
    for (int d = 0; d < dim; d ++)
    {
        // get adjacents in each dimension
        dc[d] = 1;
        if (adjacent_index(index, dc, adj_index))
            total_variation_bw(index, adj_index, dL_dtv);
        dc[d] = -1;
        if (adjacent_index(index, dc, adj_index))
            total_variation_bw(index, adj_index, dL_dtv);
        dc[d] = 0;
    }
}