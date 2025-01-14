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
#include "vr_path_integrator_NEE_DS.h"

//FORWARD
//{
//    vec3 x = vec3(_input[0], _input[1], _input[2]);
//    vec3 w = vec3(_input[3], _input[4], _input[5]);
//
//    float T_direct = transmittance(object, x, w);
//    vec3 R_direct = T_direct * environment(object, w);
//
//    vec3 wnee;
//    vec3 env = environment_sampler(object, x, w, wnee);
//    vec3 A;
//    accumulate_path_NEE_DT_fw(object, x, w, wnee, A);
//
//    vec3 R = A * env + R_direct;
//    _output = float[] (R.x, R.y, R.z);
//}
//
//BACKWARD
//{
//    vec3 x = vec3(_input[0], _input[1], _input[2]);
//    vec3 w = vec3(_input[3], _input[4], _input[5]);
//    vec3 dL_dR = vec3(_output_grad[0], _output_grad[1], _output_grad[2]);
//
//    SAVE_SEED(main_seed);
//    SAVE_SEED(before_direct_T);
//    float T_direct = transmittance(object, x, w);
//    vec3 env_direct = environment(object, w);
//
//    vec3 wnee;
//    vec3 env = environment_sampler(object, x, w, wnee);
//    SAVE_SEED(before_accumulate);
//    vec3 A;
//    accumulate_path_NEE_DT_fw(object, x, w, wnee, A);
//
//    float dL_dTdirect = dot(dL_dR, env_direct);
//    USING_SECONDARY_SEED(main_seed, before_direct_T, transmittance_bw(object, x, w, dL_dTdirect));
//
//    vec3 dL_dA = dL_dR * env;
//    USING_SECONDARY_SEED(main_seed, before_accumulate, accumulate_path_NEE_DT_bw(object, x, w, wnee, A, dL_dA));
//}



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
    path_integrator_NEE_DS_bw(object, x, w, parameters.ds_epsilon, dL_dR);
}