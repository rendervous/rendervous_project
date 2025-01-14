// Collision Integrator: <I> = \int_0^d T(t)sigma(t) F(t) dt + T(d) B
// requires: boundary, collision_sampler, exitance_radiance, boundary_radiance
// input: x, w
// ouput: <I>

#include "vr_collision_sampler.h"
#include "vr_exitance_radiance.h"
#include "vr_environment.h"

FORWARD
{
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);

    float T, t;
    float C = collision_sampler(object, x, w, t, T);
    vec3 xt = w * t + x;
    vec3 R = C * exitance_radiance(object, xt, -w) + T * environment(object, w);

    _output = float[] (R.x, R.y, R.z);
}

BACKWARD
{
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    vec3 dL_dR = vec3(_output_grad[0], _output_grad[1], _output_grad[2]);

    float T, t;
    float C = collision_sampler(object, x, w, t, T);
    vec3 xt = w * t + x;
    vec3 eR = exitance_radiance(object, xt, -w);
    vec3 env = environment(object, w);
    //return C * exitance_radiance(object, xt, -w) + T * environment(object, w);

    vec3 dL_dexitance = dL_dR * C;
    float dL_dC = dot(dL_dR, eR);
    float dL_dT = dot(dL_dR, env);
    vec3 dL_denv = dL_dR * T;

    collision_sampler_bw(object, x, w, dL_dC, dL_dT);
    environment_bw(object, w, dL_denv);
    exitance_radiance_bw(object, xt, -w, dL_dexitance);
}