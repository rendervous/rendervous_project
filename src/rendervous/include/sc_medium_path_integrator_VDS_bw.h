// Backward operation for Volume Delta-tracking with Defensive Sampling strategy

void medium_path_integrator_bw(map_object, vec3 x, vec3 w, float d, vec3 dL_dW, vec3 dL_dA)
{
    SAVE_SEED(before_path);
    vec3 xini = x;
    vec3 wini = w;
    float dini = d;
    float EPS = 1e-8; // Epsilon used as threshold to assume a singular hit
    float singular_W = 1.0; // float to consider specially singular bounces
    vec3 W = vec3(1.0);
    vec3 A = vec3(0.0);
    vec3 sA = vec3(0.0);
    float epsilon = parameters.ds_epsilon;
    float majorant = max(0.00001, param_float(parameters.majorant) + 2 * epsilon);
    while(true)
    {
        float t = -log(1 - random()) / majorant;
        if (t > d - 0.00001) // exit
            break;
        x += t * w;
        d -= t;
        float s = sigma(object, x);
        float Pc = (s + epsilon) / majorant;
        if (random() < Pc) // interaction
        {
            if (s < EPS) // singular hit
            {
                if (singular_W < 1.0) // second singular hit
                break;
                // else first-time singular hit
                singular_W = 0.0;
            }
            else {
                W *= s / majorant / Pc;
            }
    #if MEDIUM_FILTER & 2  // Accumulation is required
            vec3 G = W * gathering(object, x, w);
            if (singular_W == 1.0)
                A += G;
            else
                sA += G;
    #endif
    #if MEDIUM_FILTER & 1  // Scattering is required
            float weight, pdf_wo;
            w = sample_phase(object, w, weight, pdf_wo);
            W *= scattering_albedo(object, x) * weight;
            // recompute distance towards new direction w
            if (!raycast(object, x, w, d))
            break; // should never leave this way...
    #else
            W = vec3(0.0);
            break;
    #endif
        }
        else
        {
            W *= (1 - s / majorant) / (1 - Pc);
        }
    }

    // Replay
    SET_SEED(before_path);
    x = xini;
    w = wini;
    d = dini;
    vec3 rem_A = A;
    vec3 outW = W;
    W = vec3(1.0);

    while(true)
    {
        float t = -log(1 - random()) / majorant;
        if (t > d - 0.00001) // exit
            break;
        x += t * w;
        d -= t;
        float s = sigma(object, x);
        float Pc = (s + epsilon) / majorant;
        if (random() < Pc) // interaction
        {
            if (s >= EPS)
            {
                W *= s / majorant / Pc;

    #if MEDIUM_FILTER & 2  // Accumulation is required
                SAVE_SEED(G)
                vec3 G = gathering(object, x, w);
                SAVE_SEED(after_G)
                SET_SEED(G)
                gathering_bw(object, x, w, G, W * dL_dA);
                SET_SEED(after_G)
    #endif
    #if MEDIUM_FILTER & 1  // Scattering is required
                float weight, rho;
                w = sample_phase(object, w, weight, rho);
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
            else {
                float dL_ds = dot(dL_dA, sA);
                if (dL_ds != 0)
                    sigma_bw(object, x, dL_ds);
                break; // after singular hit no more gradient is backprop
            }
        }
        else
        {
            float dL_ds = 0;
            dL_ds += -dot(dL_dW, outW) / majorant / (1 - s/majorant);
    #if MEDIUM_FILTER & 2 == 2
            dL_ds += -dot(dL_dA, rem_A) / majorant / (1 - s/majorant);
    #endif
            if (dL_ds != 0)
                sigma_bw(object, x, dL_ds);
            W *= (1 - s / majorant) / (1 - Pc);
        }
    }
}
