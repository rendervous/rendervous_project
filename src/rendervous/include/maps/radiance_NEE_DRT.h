// Radiance via Delta-tracking
// requires: sigma, scattering_albedo, emission, boundary, majorant, environment, phase_sampler

#include "vr_sigma.h"
#include "vr_scattering_albedo.h"
#include "vr_emission.h"
#include "vr_environment.h"
#include "vr_environment_sampler.h"
#include "vr_phase.h"
#include "vr_phase_sampler.h"
#include "vr_boundary.h"
#include "vr_majorant.h"
#include "vr_transmittance.h"
#include "vr_path_integrator_NEE.h"
#include "vr_path_integrator_NEE_DRT.h"

FORWARD
{
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    vec3 R = path_integrator_NEE(object, x, w, false);
    _output = float[] (R.x, R.y, R.z);
}

BACKWARD
{
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    vec3 dL_dR = vec3(_output_grad[0], _output_grad[1], _output_grad[2]);
    path_integrator_NEE_DRT_bw(object, x, w, dL_dR);
}