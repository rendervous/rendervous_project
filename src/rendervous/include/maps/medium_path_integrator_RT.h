// This code is an implementation of a medium path integrator
// Ratio-Tracking integrator for heterogeneous cases
#include "medium_path_integrator_interface.h"

// requirements
#include "vr_sigma.h"
#include "vr_scattering_albedo.h"
#include "sc_phase_sampler.h"

void medium_path_integrator(map_object, vec3 x, vec3 w, float d, out float T, out vec3 xo, out vec3 wo, out vec3 W, out vec3 A)
{
    T = 1.0;
    W = vec3(0.0);
    A = vec3(0.0);
    float majorant = max(0.00001, param_float(parameters.majorant));
    while(true)
    {
        float t = -log(1 - random()) / majorant;
        if (t > d - 0.00001) // exit
            break;
        x += t * w;
        d -= t;
        float s = sigma(object, x);
    #if MEDIUM_FILTER != 0
        if (random() < s / majorant) // interaction
        {
            T = 0.0;
    #if MEDIUM_FILTER & 1 == 1
            xo = x;
            float weight, rho;
            wo = sample_phase(object, w, weight, rho);
            W = scattering_albedo(object, x);
    #endif
    #if MEDIUM_FILTER & 2 == 2
            A = gathering(object, x, w);
    #endif
            return;
        }
    #else
        T *= (1 - s/majorant);
        if (T < 0.01)
        {
            if (random() >= T)
            {
                T = 0.0;
                return;
            }
            T = 1.0;
        }
    #endif
    }
}

void medium_path_integrator_bw(map_object, vec3 x, vec3 w, float d, float outT, vec3 outW, vec3 outA, float dL_dT, vec3 dL_dW, vec3 dL_dA)
{
    float majorant = max(0.00001, param_float(parameters.majorant));
    while(true)
    {
        float t = -log(1 - random()) / majorant;
        if (t > d - 0.00001) // exit
            break;
        x += t * w;
        d -= t;
        float s = sigma(object, x);
        float Pc = s / majorant;
        if (random() < Pc) // interaction
        {
            /*Tr = 0.0;*/
            // transmittance 0 needs no backprop

            // backprop to sigma dL_dA and dL_dW
    #if MEDIUM_FILTER & 1 == 1
            /* xo = x;
            float weight, rho;
            wo = sample_phase(object, w, weight, rho);
            W = scattering_albedo(object, x);*/
            scattering_albedo_bw(object, x, dL_dW);
    #endif
    #if MEDIUM_FILTER & 2 == 2
            /*A = gathering(object, x, w);*/
            gathering_bw(object, x, w, dL_dA);
    #endif
            break;
        }
        // if no interaction backprop to sigma dL_dT
        if (outT > 0)
            sigma_bw(object, x, -outT * dL_dT/majorant/(1 - Pc));
    }
}
