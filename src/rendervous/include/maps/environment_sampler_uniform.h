#include "environment_sampler_interface.h"

#include "sc_environment.h"

void environment_sampler(map_object, vec3 x, out vec3 w, out vec3 E, out float pdf)
{
    w = randomDirection();
    environment(object, x, w, E);
    pdf = .25/pi;
}

void environment_sampler_bw(map_object, vec3 x, vec3 out_w, vec3 out_E, vec3 dL_dw, vec3 dL_dE)
{
    environment_bw(object, x, out_w, dL_dE);
}
