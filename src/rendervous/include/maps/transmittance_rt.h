// Transmittance via Ratio-tracking
// requires: sigma, boundary, majorant

#include "vr_sigma.h"
#include "vr_boundary.h"
#include "vr_majorant.h"
#include "vr_transmittance_RT.h"

FORWARD
{
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);

    _output[0] = transmittance_RT(object, x, w);
}

BACKWARD
{
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    float dL_dT = _output_grad[0];

    transmittance_RT_bw(object, x, w, dL_dT);
}