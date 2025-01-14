#version 460
#extension GL_GOOGLE_include_directive : require
#include "../common_functions.h"

layout(binding = 1, std430) uniform Custom {
    GPUPtr in_tensor; // image or volume to compute the TV
    GPUPtr out_tensor; // same dimensions of the input but 1 channel
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

float total_variation(int index, int adj_index)
{
    int channels = shape[dim];
    float_ptr center_value = float_ptr(in_tensor + index * channels * 4);
    float_ptr adj_value = float_ptr(in_tensor + adj_index * channels * 4);
    float tv = 0.0;
    [[unroll]]
    for (int i=0; i<channels; i++)
        tv += abs(center_value.data[i] - adj_value.data[i]);
    return tv;
}

void main() {
    uvec3 thread_id;
    if (!start_function(thread_id))
    return;

    int index = int(thread_id.x);

    float tv = 0.0;
    int dc[4] = {0, 0, 0, 0};
    int adj_index;
    [[unroll]]
    for (int d = 0; d < dim; d ++)
    {
        // get adjacents in each dimension
        dc[d] = 1;
        if (adjacent_index(index, dc, adj_index))
            tv += total_variation(index, adj_index);
        dc[d] = -1;
        if (adjacent_index(index, dc, adj_index))
            tv += total_variation(index, adj_index);
        dc[d] = 0;
    }

    float_ptr out_buf = float_ptr(out_tensor + index * 4);
    out_buf.data[0] = tv;
}