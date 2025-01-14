// Transmittance via Regular Grid Ratiotracking with Decomposition tracking
// requires: grid (sigmas), box_min, box_max, boundary

#include "vr_boundary.h"
#include "vr_sigma_grid.h"
#include "vr_exitance_radiance.h"
#include "vr_environment.h"
#include "vr_collision_integrator_DDA.h"

FORWARD
{
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    vec3 R = collision_integrator_DDA(object, x, w);
    _output = float[](R.x, R.y, R.z);
}

BACKWARD
{
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    vec3 dL_dR = vec3(_output_grad[0], _output_grad[1], _output_grad[2]);

    collision_integrator_DDA_bw(object, x, w, dL_dR);
}