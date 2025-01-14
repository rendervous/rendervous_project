#include "environment_interface.h"

void environment(map_object, vec3 x, vec3 w, out vec3 E)
{
    vec2 xr = dir2xr(w);
    float[3] R;
    forward(parameters.environment_img, float[2](xr.x, xr.y), R);
    E = vec3(R[0], R[1], R[2]);
}

void environment_bw(map_object, vec3 x, vec3 w, vec3 dL_dE)
{
    vec2 xr = dir2xr(w);
    float[3] dL_dR = float[3](dL_dE.x, dL_dE.y, dL_dE.z);
    backward(parameters.environment_img, float[2](xr.x, xr.y), dL_dR);
}