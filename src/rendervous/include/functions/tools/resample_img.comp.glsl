#version 460
#extension GL_GOOGLE_include_directive : require
#include "../common_functions.h"

layout(binding = 1, std430) uniform Custom {
    float_ptr in_tensor;
    float_ptr out_tensor;
    int[4] in_shape;
    int[4] out_shape;
    int dim;
    int output_dim;
};


void thread_idx_to_coordinates(int index, out int in_coord[4], out float in_alpha[4])
{
    [[unroll]]
    for (int i=0; i<dim; i++)
    {
        int id = in_shape[dim-i-1];
        int od = out_shape[dim-i-1];
        int pc = index % od;
        float coord = (pc + 0.5)/float(od) * id - 0.5;
        in_coord[i] = int(floor(coord));
        in_alpha[i] = coord - in_coord[i];
        index /= od;
    }
}


void load_and_blend(int out_index, int in_coord[4], int in_dx[4], float alpha)
{
    int index = 0;
    int u = 1;
    [[unroll]]
    for (int i=0; i<dim; i++)
    {
        int d = in_shape[dim - i - 1];
        int c = max(0, min(in_coord[i] + in_dx[i], d - 1));
        index += c * u;
        u *= d;
    }
    [[unroll]]
    for (int i=0; i<output_dim; i++)
        out_tensor.data[out_index * output_dim + i] += alpha * in_tensor.data[index * output_dim + i];
}


void main() {
    uvec3 thread_id;
    if (!start_function(thread_id))
    return;

    int index = int(thread_id.x);

    int in_coord[4];
    float in_alpha[4];
    thread_idx_to_coordinates(index, in_coord, in_alpha); // get the

    [[unroll]]
    for (int i=0; i<output_dim; i++)
        out_tensor.data[index * output_dim + i] = 0.0; // clean the output at index

    switch (dim)
    {
        case 1:
        load_and_blend(index, in_coord, int[](0,0,0,0), (1 - in_alpha[0]));
        load_and_blend(index, in_coord, int[](1,0,0,0), (in_alpha[0]));
        break;
        case 2:
        load_and_blend(index, in_coord, int[](0,0,0,0), (1 - in_alpha[0])*(1 - in_alpha[1]));
        load_and_blend(index, in_coord, int[](1,0,0,0), (in_alpha[0])*(1 - in_alpha[1]));
        load_and_blend(index, in_coord, int[](0,1,0,0), (1 - in_alpha[0])*(in_alpha[1]));
        load_and_blend(index, in_coord, int[](1,1,0,0), (in_alpha[0])*(in_alpha[1]));
        break;
        case 3:
        load_and_blend(index, in_coord, int[](0,0,0,0), (1 - in_alpha[0])*(1 - in_alpha[1])*(1 - in_alpha[2]));
        load_and_blend(index, in_coord, int[](1,0,0,0), (in_alpha[0])*(1 - in_alpha[1])*(1 - in_alpha[2]));
        load_and_blend(index, in_coord, int[](0,1,0,0), (1 - in_alpha[0])*(in_alpha[1])*(1 - in_alpha[2]));
        load_and_blend(index, in_coord, int[](1,1,0,0), (in_alpha[0])*(in_alpha[1])*(1 - in_alpha[2]));
        load_and_blend(index, in_coord, int[](0,0,1,0), (1 - in_alpha[0])*(1 - in_alpha[1])*(in_alpha[2]));
        load_and_blend(index, in_coord, int[](1,0,1,0), (in_alpha[0])*(1 - in_alpha[1])*(in_alpha[2]));
        load_and_blend(index, in_coord, int[](0,1,1,0), (1 - in_alpha[0])*(in_alpha[1])*(in_alpha[2]));
        load_and_blend(index, in_coord, int[](1,1,1,0), (in_alpha[0])*(in_alpha[1])*(in_alpha[2]));
        break;
    }
}