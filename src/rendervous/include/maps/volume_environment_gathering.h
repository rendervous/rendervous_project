#include "volume_gathering_interface.h"
#include "vr_scattering_albedo.h"
#include "vr_homogeneous_phase.h"
#include "sc_visibility.h"
#include "sc_environment_sampler.h"


void volume_gathering(map_object, vec3 x, vec3 w, out vec3 A)
{
    A = vec3(0.0);

    vec3 we, E;
    float pdf;
    environment_sampler(object, x, we, E, pdf);

//    if (E == vec3(0.0))
//        return;

    vec3 W = scattering_albedo(object, x);
//    if (W == vec3(0.0))
//        return;

    float rho = homogeneous_phase(object, w, we);
//    if (rho == 0)
//        return;

    float Tr = ray_visibility(object, x, we);
    A = Tr * W * rho * E;
}

void volume_gathering_bw(map_object, vec3 x, vec3 w, vec3 out_A, vec3 dL_dA)
{
    //if (dL_dA == vec3(0.0))
    //return;

    vec3 we, E;
    float pdf;
    environment_sampler(object, x, we, E, pdf);
    vec3 W = scattering_albedo(object, x);
    float rho = homogeneous_phase(object, w, we);
    SAVE_SEED(before_Tr);
    float Tr = ray_visibility(object, x, we);
    SAVE_SEED(after_Tr);
    // use the output Accumulate to avoid repeating visibility
    //vec3 Tr = out_A/(W * E * rho);//ray_visibility(object, x, we);
    //Tr = mix (Tr, vec3(0.0), isnan(Tr));
    //A += Tr * W * rho * E;
    float dL_dTr = dot(dL_dA, W * rho * E);
    vec3 dL_dE = dL_dA * W * rho * Tr;
    vec3 dL_dW = dL_dA * Tr * rho * E;
    float dL_drho = dot(dL_dA, Tr * W * E);
    scattering_albedo_bw(object, x, dL_dW);
    //ray_visibility_bw(object, x, we, max(Tr.x, max(Tr.y, Tr.z)), dL_dTr);
    SET_SEED(before_Tr);
    ray_visibility_bw(object, x, we, Tr, dL_dTr);
    ASSERT(get_seed() == SEED(after_Tr), "volume_environment_gathering.h: Seed inconsistency.")
    SET_SEED(after_Tr);
    homogeneous_phase_bw(object, w, we, dL_drho);
    //environment_sampler_bw(object, x, we, E, dL_dE); // TODO: not supported yet
}
