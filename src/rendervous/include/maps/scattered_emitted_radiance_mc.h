#include "vr_phase_sampler.h"
#include "vr_scattering_albedo.h"
#include "vr_emission.h"
#include "vr_radiance.h"

FORWARD {
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);

    vec3 wo;
    float ph = phase_sampler(object, x, -w, wo);
    vec3 sa = scattering_albedo(object, x);
    vec3 em = emission(object, x, -w);
    vec3 R = sa * ph * radiance(object, x, wo) + (vec3(1.0) - sa) * em;
    _output = float[](R.x, R.y, R.z);
}

BACKWARD
{
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    vec3 dL_dR = vec3(_output_grad[0], _output_grad[1], _output_grad[2]);

    vec3 wo;
    float ph = phase_sampler(object, x, -w, wo);
    vec3 sa = scattering_albedo(object, x);
    vec3 rad = radiance(object, x, wo);
    vec3 em = emission(object, x, -w);

    //return sa * ph * radiance(object, x, wo);
    float dL_dph = dot(dL_dR, sa*rad);
    vec3 dL_dsa = dL_dR * ph * rad - em;
    vec3 dL_drad = dL_dR * sa * ph;
    vec3 dL_dem = dL_dR * (vec3(1.0) - sa);

    phase_sampler_bw(object, x, -w, dL_dph);
    scattering_albedo_bw(object, x, dL_dsa);
    radiance_bw(object, x, w, dL_dR);
    emission_bw(object, x, -w, dL_dem);
}

