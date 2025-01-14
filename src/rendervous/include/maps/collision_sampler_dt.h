// Collision Sampling via Delta-tracking
// requires: sigma, boundary, majorant
// input: x, w
// ouput: <T(d)>, t, w(t) ~ T(t) sigma(t)

#include "vr_sigma.h"
#include "vr_boundary.h"
#include "vr_majorant.h"
#include "vr_collision_sampler_DT.h"

FORWARD
{
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    _output[0] = collision_sampler_DT(object, x, w, parameters.ds_epsilon, _output[1], _output[2]);
}

BACKWARD
{
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    float dL_dC = _output_grad[0];
    float dL_dT = _output_grad[2];
    collision_sampler_DT_bw(object, x, w, parameters.ds_epsilon, dL_dC, dL_dT);
}