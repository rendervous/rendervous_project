// This code is an implementation of a medium path integrator
// Delta-Tracking integrator for heterogeneous cases
#include "medium_path_integrator_interface.h"

// requirements
#include "vr_sigma.h"
#include "vr_scattering_albedo.h"
#include "sc_phase_sampler.h"

void medium_path_integrator(map_object, vec3 x, vec3 w, float d, out vec3 xo, out vec3 wo, out vec3 W, out vec3 A)
{
    W = vec3(1.0);
    A = vec3(0.0);
    xo = d * w + x;
    wo = w;
    float epsilon = 1.0;
    float majorant = max(0.00001, param_float(parameters.majorant) + 2 * epsilon);
    while(true)
    {
        float t = -log(1 - random()) / majorant;
        if (t > d - 0.00001) // exit
            break;
        x += t * w;
        d -= t;
        float s = sigma(object, x) + 1e-10;
        float Pc = (s + epsilon) / majorant;
        if (random() < Pc) // interaction
        {
            W *= s / majorant / Pc;
    #if MEDIUM_FILTER & 2  // Accumulation is required
            A = W * gathering(object, x, w);
    #endif
    #if MEDIUM_FILTER & 1  // Scattering is required
            xo = x;
            float weight, rho;
            wo = sample_phase(object, w, weight, rho);
            W *= scattering_albedo(object, x);
    #else
            W = vec3(0.0);
    #endif
            break;
        }
        else
        {
            W *= (1 - s / majorant) / (1 - Pc);
        }
    }
}

void medium_path_integrator_bw(map_object, vec3 x, vec3 w, float d, vec3 outW, vec3 outA, vec3 dL_dW, vec3 dL_dA)
{
    float epsilon = 1.0;
    float majorant = max(0.00001, param_float(parameters.majorant) + 2 * epsilon);
    vec3 W = vec3(1.0);

    while(true)
    {
        float t = -log(1 - random()) / majorant;
        if (t > d - 0.00001) // exit
            break;
        x += t * w;
        d -= t;
        float s = sigma(object, x) + 1e-10;
        float Pc = (s + epsilon) / majorant;
        if (random() < Pc) // interaction
        {
            W *= s / majorant / Pc;

            /*Tr = 0.0;*/
            // transmittance 0 needs no backprop

            // backprop to sigma dL_dA and dL_dW
            float dL_dsigma = 0;
    #if MEDIUM_FILTER & 2 == 2
            /*A = gathering(object, x, w);*/
            //vec3 dA_dW = outA / W;
            //vec3 dW_ds = W / s;
            dL_dsigma += dot(dL_dA, outA / s);
            gathering_bw(object, x, w, outA / W, dL_dA * W);
    #endif
    #if MEDIUM_FILTER & 1 == 1
            /* xo = x;
            float weight, rho;
            wo = sample_phase(object, w, weight, rho);
            W = scattering_albedo(object, x);*/
            float weight, rho;
            sample_phase(object, w, weight, rho); // only necessary to replay same randoms
            dL_dsigma += dot(dL_dW, outW / s);
            scattering_albedo_bw(object, x, dL_dW);
    #endif

            if (dL_dsigma != 0)
                sigma_bw(object, x, dL_dsigma);
            break;
        }
        // if no interaction backprop to sigma dL_dT, dL_dW, dL_dA
        else
        {
            float dL_dsigma = 0;
            dL_dsigma += -dot(dL_dW, outW) / majorant / (1 - s/majorant);
    #if MEDIUM_FILTER & 2 == 2
            dL_dsigma += -dot(dL_dA, outA) / majorant / (1 - s/majorant);
    #endif
            if (dL_dsigma != 0)
                sigma_bw(object, x, dL_dsigma);

            W *= (1 - s / majorant) / (1 - Pc);
        }
        //sigma_bw(object, x, (-outT * dL_dT + dot((1-outT)* dL_dW, vec3(1.0)) + dot((1-outT)*dL_dA, vec3(1.0)))/majorant/(1 - Pc));
    }
}
