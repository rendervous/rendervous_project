// Transmittance via Regular Grid Ratiotracking with Decomposition tracking
// requires: grid (sigmas), box_min, box_max, boundary

#include "vr_boundary.h"
#include "vr_sigma_grid.h"
#include "vr_transmittance_DDA.h"

FORWARD
{
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    _output[0] = transmittance_DDA(object, x, w);
}

BACKWARD
{
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    float dL_dT = _output_grad[0];

    if (sigma_grid_has_grad(object))
        transmittance_DDA_bw(object, x, w, dL_dT);
}