#include "medium_path_integrator_interface.h"

// requirements
#include "vr_sigma.h"
#include "vr_scattering_albedo.h"
#include "sc_phase_sampler.h"
#include "sc_boundaries.h"
#include "sc_medium_transmittance_RT.h"


bool sample_collision_position(map_object,
    vec3 x, vec3 w, float d,
    out vec3 xc, // sampled point
    out float weight_xc, // weight of the sampling  ~T or ~Tsigma if singular
    out float sigma_xc, // sigma at collision point
    out int samples, // used to resample path segment in backward with same frequency
    inout bool mis_allowed // indicates if MIS is used or not, out as false if transmittance was used
    ){
    float MIS_BETA = 3;
    float t = 0;
    weight_xc = 0.0;
    sigma_xc = 0.0;
    samples = 0;
    float m = parameters.majorant + 0.000001; // to avoid creating outliers with s / m

    while (true) {
        float dt = min(-log(1 - random()) / m, d - t);
        t += dt;
        xc = w * t + x;
        if (t >= d - 0.0000001)
        break; // endpoint reached
        float s = sigma(object, xc);
        samples++;
        float Ps = s / m;
        if (random() < Ps) // real collision
        {
            weight_xc = 1.0;
            sigma_xc = s;
            break;
        }
    }

    if (!mis_allowed)
    return false;

    float powered_c = pow(sigma_xc, MIS_BETA);

    // mis weight for the collision (weighted again wrt sampler)
    float mis_wc = weight_xc * powered_c / (1 + powered_c);

    float weight_f = t; // normalized pdf
    //float sample_f = (1 - random()) * t;
    //vec3 xf = x + w * sample_f;
    vec3 xf = mix(xc, x, random());
    float sigma_f = sigma(object, xf);
    float powered_f = pow(sigma_f, MIS_BETA);
    float mis_wf = t / (1 + powered_f);

    weight_xc = (mis_wc + mis_wf);
    if (random() * weight_xc < mis_wf) // bounce in potential zero region
    {
        //weight_xc *= weight_f;
        xc = xf;
        sigma_xc = sigma_f;
        mis_allowed = false; // mark the singular vertex was already hit
        return true; // singular position was hit
    }

    return false; // was no singular hit, check weight_c to know if hit at all.
}


// Method used to backprop tau (optical depth) inside a segment
void st_tau_bw(map_object, vec3 a, vec3 b, int bins, float dL_dtau)
{
    if (dL_dtau == 0)
    return;

    if (bins == 0)
    return;

    float d = length(a - b);
    float dL_dsigma = dL_dtau * d / bins;

    [[unroll]]
    for (int i=0; i<bins; i++)
        sigma_bw(object, mix(b, a, (i+random())/bins), dL_dsigma);
}


void medium_path_integrator(map_object, vec3 x, vec3 w, float d, out vec3 xo, out vec3 wo, out vec3 W, out vec3 A)
{
    // Consider radiance seen direct through
    if (random() < 0.5)
    {
        float direct_T = transmittance(object, x, w, d); // No scatter case handled appart
        xo = w * d + x;
        wo = w;
        W = vec3(2.0 * direct_T);
        A = vec3(0.0);
        return;
    }

    // Accumulated radiance along the primary path
    A = vec3(0); // NEE accumulation before the singular vertex
    vec3 Af = vec3(0); // NEE accumulation from singular vertex, without singular factor
    vec3 W = vec3(1); // Path throughput without factor
    float W_singular = 1.0; // Path throughput factor at singular vertex

    bool mis_allowed = true;
    vec3 ini_x = x;
    vec3 ini_w = w;
    float ini_d = d;

    float EPS = 0.00000001;

    SAVE_SEED(before_path);

    while (true) {
        // Sample Collision and Transmittance
        float weight_xc;
        vec3 xc; float sigma_xc;
        int samples;
        if (sample_collision_position( // Sample a collision with MIS between DT and DRT. Returns if sample was possible. If mis_allowed if false, DT is used always.
                object,
                x, w, d,
                xc, weight_xc, sigma_xc, samples, mis_allowed))
                W_singular = sigma_xc; // singular vertex

        if (weight_xc == 0.0) // no collision
        break;

        x = xc;
        W *= weight_xc * scattering_albedo(object, x);

        // sample NEE direction
        vec3 wnee;
        vec3 nee_env = environment_sampler(object, x, w, wnee);
        float nee_T = transmittance(object, x, wnee);
        float nee_rho = phase(object, x, w, wnee);
        vec3 added_R = W * nee_T * nee_rho * nee_env;
        R += int(mis_allowed) * added_R;
        Rf += int(!mis_allowed) * added_R;

        // continue path
        vec3 wo;
        float rho = phase_sampler(object, x, w, wo);
        W *= rho;
        w = wo;
        boundary(object, x, w, tMin, d);

        if (all(lessThan(W, vec3(EPS))))
        break;
    }
}
