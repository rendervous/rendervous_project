#include "vr_phase.h"
#include "vr_environment_sampler.h"
#include "vr_scattering_albedo.h"
#include "vr_transmittance.h"

FORWARD {
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);

    vec3 wo;
    vec3 env = environment_sampler(object, x, w, wo);
    float ph = phase(object, x, -w, wo);
    vec3 sa = scattering_albedo(object, x);
    vec3 R = env * sa * ph * transmittance(object, x, wo);
    _output = float[](R.x, R.y, R.z);
}

BACKWARD
{
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    vec3 dL_dR = vec3(_output_grad[0], _output_grad[1], _output_grad[2]);

    vec3 wo;
    vec3 env = environment_sampler(object, x, w, wo);
    float ph = phase(object, x, -w, wo);
    vec3 sa = scattering_albedo(object, x);
    float T = transmittance(object, x, wo);

    //return sa * ph * radiance(object, x, wo);
    float dL_dph = dot(dL_dR, env * sa * T);
    vec3 dL_dsa = dL_dR * env * ph * T;
    float dL_dT = dot(dL_dR, env * sa * ph);

    phase_bw(object, x, -w, wo, dL_dph);
    scattering_albedo_bw(object, x, dL_dsa);
    transmittance_bw(object, x, w, dL_dT);
}

