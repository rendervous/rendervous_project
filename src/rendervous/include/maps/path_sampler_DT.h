/*
For a given ray x,w samples an ending state xo, wo, W
if xo is within volume, the path was absorved, if ray leaves the volume xo is granted to be outside far away
*/
#include "vr_sigma.h"
#include "vr_scattering_albedo.h"
#include "vr_phase_sampler.h"
#include "vr_boundary.h"
#include "vr_majorant.h"
#include "vr_path_sampler_DT.h"
#include "vr_environment.h"

FORWARD
{
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);

    vec3 wo, W;
    path_sampler_DT(object, x, w, wo, W);

    vec3 E = environment(object, wo);

    vec3 R = W * E;

    _output = float[] (R.x, R.y, R.z);
}

BACKWARD
{
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    vec3 dL_dR = vec3(_output_grad[0], _output_grad[1], _output_grad[2]);

    vec3 wo, W;
    path_sampler_DT(object, x, w, wo, W);

    vec3 E = environment(object, wo);

    //vec3 R = W * E;
    vec3 dL_dW = dL_dR * E;
    vec3 dL_dE = dL_dR * W;
    environment_bw(object, wo, dL_dE);

    // TODO: Path sampler bw to update scatter albedos!
}