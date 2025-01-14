// Volumetric Delta-tracking backward implementation without defensive sampling
void medium_path_integrator_bw(map_object, vec3 x, vec3 w, float d, vec3 dL_dW, vec3 dL_dA)
{
    float majorant = max(0.00001, param_float(parameters.majorant));

    SAVE_SEED(before_path);
    vec3 W = vec3(1.0);
    vec3 A = vec3(0.0);
    vec3 xini = x;
    vec3 wini = w;
    float dini = d;
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
    #if MEDIUM_FILTER & 2  // Accumulation is required
            A += W * gathering(object, x, w);
    #endif
    #if MEDIUM_FILTER & 1  // Scattering is required
            float weight, pdf_wo;
            w = sample_phase(object, w, weight, pdf_wo);
            W *= scattering_albedo(object, x) * weight;
            // recompute distance towards new direction w
            if (!raycast(object, x, w, d))
            break;
    #else
            W = vec3(0.0);
            break;
    #endif
        }
    }

    // Replay
    SET_SEED(before_path);
    vec3 outW = W;
    vec3 rem_A = A;
    W = vec3(1.0);
    A = vec3(0.0);
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
        float s = sigma(object, x);;
        float Pc = s / majorant;
        if (random() < Pc) // interaction
        {
    #if MEDIUM_FILTER & 2  // Accumulation is required
            SAVE_SEED(G)
            vec3 G = gathering(object, x, w);
            SET_SEED(G)
            gathering_bw(object, x, w, G, W * dL_dA);
    #endif
    #if MEDIUM_FILTER & 1  // Scattering is required
            float weight, pdf_wo;
            w = sample_phase(object, w, weight, pdf_wo);
            vec3 sa = scattering_albedo(object, x);
    #endif
            float dL_ds = (dot(dL_dA, rem_A) + dot(dL_dW, outW))/s;
            if (dL_ds != 0)
                sigma_bw(object, x, dL_ds);

    #if MEDIUM_FILTER & 2
            rem_A -= W * G;
    #endif

    #if MEDIUM_FILTER & 1
            // recompute distance towards new direction w
            W *= sa * weight;
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