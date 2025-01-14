// Transmittance via Regular Grid Ratiotracking with Decomposition tracking
// requires: grid (sigmas), box_min, box_max
#include "vr_sigma_grid.h"
#include "vr_boundary.h"
#include "vr_transmittance_GRT.h"

FORWARD
{
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);

    _output[0] = transmittance_GRT(object, x, w);
}
