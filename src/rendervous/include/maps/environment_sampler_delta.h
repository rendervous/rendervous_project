#include "environment_sampler_interface.h"

void environment_sampler(map_object, vec3 x, out vec3 w, out vec3 E, out float pdf)
{
    w = param_vec3(parameters.light_direction);
    E = param_vec3(parameters.light_intensity);
    pdf = 1.0;
}

void environment_sampler_bw(map_object, vec3 x, vec3 out_w, vec3 out_E, vec3 dL_dw, vec3 dL_dE)
{
    param_grad_vec3(parameters.light_direction, dL_dw);
    param_grad_vec3(parameters.light_intensity, dL_dE);
}
