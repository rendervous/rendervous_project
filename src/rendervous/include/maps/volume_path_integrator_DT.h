#include "medium_path_integrator_interface.h"

// requirements
#include "vr_sigma.h"
#include "vr_scattering_albedo.h"
#include "sc_phase_sampler.h"
#include "sc_boundaries.h"

void medium_path_integrator(map_object, vec3 x, vec3 w, float d, out vec3 xo, out vec3 wo, out vec3 W, out vec3 A)
{
    W = vec3(1.0);
    A = vec3(0.0);
    float majorant = max(0.00001, param_float(parameters.majorant));
    while(true)
    {
        float t = -log(1 - random()) / majorant;
        if (t > d - 0.00001) // exit
            break;
        x += t * w;
        d -= t;
        float s = sigma(object, x);// + 1e-10;
        float Pc = s / majorant;
        if (random() < Pc) // interaction
        {
            //W *= s / majorant / Pc;
    #if MEDIUM_FILTER & 2  // Accumulation is required
            A += W * gathering(object, x, w);
    #endif
    #if MEDIUM_FILTER & 1  // Scattering is required
            float weight, rho;
            w = sample_phase(object, w, weight, rho);
            W *= scattering_albedo(object, x);
            // recompute distance towards new direction w
            if (!raycast(object, x, w, d))
            break;
    #else
            W = vec3(0.0);
            break;
    #endif
        }
        else
        {
            //W *= (1 - s / majorant) / (1 - Pc);
        }
    }
    wo = w;
    xo = w*d + x;
}

void medium_path_integrator_bw(map_object, vec3 x, vec3 w, float d, vec3 outW, vec3 outA, vec3 dL_dW, vec3 dL_dA)
{
    vec3 W = vec3(1.0);
    vec3 rem_A = outA;
    float majorant = max(0.00001, param_float(parameters.majorant));
    while(true)
    {
        float t = -log(1 - random()) / majorant;
        if (t > d - 0.00001) // exit
            break;
        x += t * w;
        d -= t;
        float s = sigma(object, x);// + 1e-10;
        float Pc = s / majorant;
        if (random() < Pc) // interaction
        {
            //W *= s / majorant / Pc;
    #if MEDIUM_FILTER & 2  // Accumulation is required
            SAVE_SEED(G)
            vec3 G = gathering(object, x, w);
            SET_SEED(G)
            gathering_bw(object, x, w, G, W * dL_dA);
    #endif
    #if MEDIUM_FILTER & 1  // Scattering is required
            float weight, rho;
            w = sample_phase(object, w, weight, rho);
            vec3 sa = scattering_albedo(object, x);
    #endif

            // Replay bw
            float dL_ds = (dot(dL_dA, rem_A) + dot(dL_dW, outW))/s;
            if (dL_ds != 0)
                sigma_bw(object, x, dL_ds);

    #if MEDIUM_FILTER & 2
            rem_A -= W * G;
    #endif

    #if MEDIUM_FILTER & 1
            // recompute distance towards new direction w
            W *= sa;
            if (!raycast(object, x, w, d))
            break;
    #else
            W = vec3(0.0);
            break;
    #endif
        }
        else
        {
            float dL_ds = 0;
            dL_ds += -dot(dL_dW, outW) / majorant / (1 - Pc);
    #if MEDIUM_FILTER & 2 == 2
            dL_ds += -dot(dL_dA, rem_A) / majorant / (1 - Pc);
    #endif
            if (dL_ds != 0)
                sigma_bw(object, x, dL_ds);
        }
    }
}