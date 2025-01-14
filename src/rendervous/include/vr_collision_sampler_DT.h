/*
Given a ray x,w will sample a position wrt T(x+wt)sigma(x+wt).
Return an estimator for the pdf, the sampled position and the transmittance.
*/
float collision_sampler_DT(map_object, vec3 x, vec3 w, float epsilon, out float out_t, out float out_T)
{
    // TODO: IMPLEMENT DEFENSIVE SAMPLING, is biased now.

    out_T = 1.0;
    out_t = 0.0;
    float C = 0.0;

    float tMin, tMax;
    if (!boundary(object, x, w, tMin, tMax))
    return C;

    float d = tMax - tMin;

    x += w * tMin;

    float t = 0; // t inside volume

    while(true){
        float md;
        float maj = majorant(object, x, w, md);
        float dt = min(-log(1 - random()) / maj, min(d, md));
        x += w * dt;
        t += dt;
        if (dt >= md)
        continue;

        if (t >= d - 0.0000001)
        break;

        if (random() < sigma(object, x)/maj)
        {
            out_T = 0.0;
            C = 1.0;
            break;
        }
    }
    out_t = tMin + t;
    return C;
}

void collision_sampler_DT_bw(map_object, vec3 x, vec3 w, float epsilon, float dL_dC, float dL_dT)
{
    float tMin, tMax;
    if (!boundary(object, x, w, tMin, tMax))
    return;

    uvec4 seed_before_cs = get_seed();

    float out_T = 1.0;
    float out_t = 0.0;
    float C = 0.0;
    float s_t = 0.0;

    float d = tMax - tMin;

    x += w * tMin;

    vec3 ini_x = x;

    float t = 0; // t inside volume

    while(true){
        float md;
        float maj = max(0.0000000001, majorant(object, x, w, md) + 2*epsilon);
        float dt = min(-log(1 - random()) / maj, min(d, md));
        x += w * dt;
        t += dt;
        if (dt >= md)
        continue;

        if (t >= d - 0.0000001)
        break;

        float s = sigma(object, x);
        float Pc = (s + epsilon) / maj;

        if (random() < Pc)
        {
            out_T = 0.0;
            C = 1.0;
            s_t = s/maj/Pc;
            break;
        }
        out_T *= (1 - s/maj)/(1 - Pc);
    }

    //out_t = tMin + t;
    //return C;
    set_seed(seed_before_cs);
    // TODO: implement replay !

    x = ini_x;
    t = 0;

    while(true){
        float md;
        float maj = max(0.0000000001, majorant(object, x, w, md) + 2*epsilon);
        float dt = min(-log(1 - random()) / maj, min(d, md));
        x += w * dt;
        t += dt;
        if (dt >= md)
        continue;

        if (t >= d - 0.0000001)
        break;

        float s = sigma(object, x);
        float Pc = (s + epsilon) / maj;

        if (random() < Pc)
        {
            float dLo_dPs = 1;
            float dPs_dgrid = 1 / maj;
            // not using fw_weight here... could potentially be 0.
            float Tr_by_sigma = C / Pc;
            float dL_dgrid = dL_dC * Tr_by_sigma * dLo_dPs * dPs_dgrid;
            sigma_bw(object, x, dL_dgrid);
            break;
        }
        else {
            //T *= (1 - sigma / majorant) / (1 - Ps);
            float dLo_dPn = 1;
            float dPn_dgrid = -1.0 / maj;
            float Tr_by_one_minus_sigma = (dL_dT * out_T + dL_dC * C * s_t) / ((1 - s / maj) / (1 - Pc)) / (1 - Pc);
            float dL_dgrid = Tr_by_one_minus_sigma * dLo_dPn * dPn_dgrid;
            sigma_bw(object, x, dL_dgrid);
        }
    }
}
