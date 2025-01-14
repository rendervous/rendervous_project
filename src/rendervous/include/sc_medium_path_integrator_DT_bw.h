void medium_path_integrator_bw(map_object, vec3 x, vec3 w, float d, vec3 dL_dW, vec3 dL_dA)
{
    vec3 W = vec3(1.0);
    vec3 A = vec3(0.0);
    // state saved for the replay
    vec3 xini = x;
    vec3 wini = w;
    float dini = d;
    SAVE_SEED(before_path);
    float majorant = max(0.00001, param_float(parameters.majorant));
    while(true)
    {
        float t = -log(1 - random()) / majorant;
        if (t > d - 0.00001) // exit
            break;
        x += t * w;
        d -= t;
        float s = sigma(object, x);
        if (random() < s / majorant) // interaction
        {
    #if MEDIUM_FILTER & 1  // Scattering is required
            float weight, pdf_wo;
            sample_phase(object, w, weight, pdf_wo);
            W = scattering_albedo(object, x) * weight;
    #else
            W = vec3(0.0);
    #endif
    #if MEDIUM_FILTER & 2  // Accumulation is required
            A = gathering(object, x, w);
    #endif
            break;
        }
    }

    // Replay path
    SET_SEED(before_path);
    x = xini;
    w = wini;
    d = dini;
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
            // backprop to sigma dL_dA and dL_dW
            float dL_dsigma = 0;
    #if MEDIUM_FILTER & 1 == 1
            /* xo = x;
            float weight, rho;
            wo = sample_phase(object, w, weight, rho);
            W = scattering_albedo(object, x);*/
            float weight, pdf_wo;
            sample_phase(object, w, weight, pdf_wo); // only necessary to replay same randoms
            dL_dsigma += dot(dL_dW, W / s);
            scattering_albedo_bw(object, x, dL_dW);
    #endif
    #if MEDIUM_FILTER & 2 == 2
            /*A = gathering(object, x, w);*/
            dL_dsigma += dot(dL_dA, A / s);
            gathering_bw(object, x, w, A, dL_dA);
    #endif
            if (dL_dsigma != 0)
                sigma_bw(object, x, dL_dsigma);
            break;
        }
        // if no interaction backprop to sigma dL_dT, dL_dW, dL_dA
        else
        {
            float dL_dsigma = 0;
            dL_dsigma += -dot(dL_dW, W) / majorant / (1 - Pc);
    #if MEDIUM_FILTER & 2 == 2
            dL_dsigma += -dot(dL_dA, A) / majorant / (1 - Pc);
    #endif
            if (dL_dsigma != 0)
                sigma_bw(object, x, dL_dsigma);
        }
        //sigma_bw(object, x, (-outT * dL_dT + dot((1-outT)* dL_dW, vec3(1.0)) + dot((1-outT)*dL_dA, vec3(1.0)))/majorant/(1 - Pc));
    }

}