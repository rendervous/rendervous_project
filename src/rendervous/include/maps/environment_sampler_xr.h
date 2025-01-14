#include "environment_sampler_interface.h"

#include "sc_environment.h"

void environment_sampler(map_object, vec3 x, out vec3 w, out vec3 E, out float pdf)
{
    float_ptr densities_buf = float_ptr(parameters.densities);
    float sel = random();
    int current_node = 0;
    vec2 p0 = vec2(0,0);
    vec2 p1 = vec2(1,1);
    float prob = 1;
    [[unroll]]
    for (int i=0; i<parameters.levels; i++)
    {
        int offset = current_node * 4;
        int selected_child = 3;
        prob = densities_buf.data[offset + 3];
        [[unroll]]
        for (int c = 0; c < 3; c ++)
            if (sel < densities_buf.data[offset + c])
            {
                selected_child = c;
                prob = densities_buf.data[offset + c];
                break;
            }
            else
                sel -= densities_buf.data[offset + c];;
        float xmed = (p1.x + p0.x)/2;
        float ymed = (p1.y + p0.y)/2;
        if (selected_child % 2 == 0) // left
            p1.x = xmed;
        else
            p0.x = xmed;
        if (selected_child / 2 == 0) // top
            p1.y = ymed;
        else
            p0.y = ymed;
        current_node = current_node * 4 + 1 + selected_child;
    }
    float pixel_area = 2 * pi * (cos(p0.y*pi) - cos(p1.y*pi)) / (1 << parameters.levels);
    w = randomDirection((p0.x * 2 - 1) * pi, (p1.x * 2 - 1) * pi, p0.y * pi, p1.y * pi);
    environment(object, x, w, E);
    E *= pixel_area / max(0.000000001, prob);
    pdf = max(0.000000001, prob) / pixel_area;
}

void environment_sampler_bw(map_object, vec3 x, vec3 out_w, vec3 out_E, vec3 dL_dw, vec3 dL_dE)
{
    // replay sampling the quadtree
    float_ptr densities_buf = float_ptr(parameters.densities);
    float sel = random();
    int current_node = 0;
    vec2 p0 = vec2(0,0);
    vec2 p1 = vec2(1,1);
    float prob = 1;
    [[unroll]]
    for (int i=0; i<parameters.levels; i++)
    {
        int offset = current_node * 4;
        int selected_child = 3;
        prob = densities_buf.data[offset + 3];
        [[unroll]]
        for (int c = 0; c < 3; c ++)
            if (sel < densities_buf.data[offset + c])
            {
                selected_child = c;
                prob = densities_buf.data[offset + c];
                break;
            }
            else
                sel -= densities_buf.data[offset + c];;
        float xmed = (p1.x + p0.x)/2;
        float ymed = (p1.y + p0.y)/2;
        if (selected_child % 2 == 0) // left
            p1.x = xmed;
        else
            p0.x = xmed;
        if (selected_child / 2 == 0) // top
            p1.y = ymed;
        else
            p0.y = ymed;
        current_node = current_node * 4 + 1 + selected_child;
    }
    float pixel_area = 2 * pi * (cos(p0.y*pi) - cos(p1.y*pi)) / (1 << parameters.levels);
    out_w = randomDirection((p0.x * 2 - 1) * pi, (p1.x * 2 - 1) * pi, p0.y * pi, p1.y * pi);
    dL_dE *= pixel_area / max(0.000000001, prob);
    environment_bw(object, x, out_w, dL_dE);
}