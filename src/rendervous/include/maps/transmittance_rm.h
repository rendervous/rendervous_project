// Transmittance via Raymarching
// requires: sigma, boundary

#include "vr_sigma.h"
#include "vr_boundary.h"
#include "vr_transmittance_RM.h"

FORWARD
{
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);

    _output[0] = transmittance_RM(object, x, w);
}