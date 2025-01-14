/*
Delta-tracking path integrator. High-variance but good as ground-truth
requires: boundary, sigma, scattering_albedo, emission, phase_sampler, phase, environment_sampler, environment, transmittance
bw requires: ds_epsilon
*/
void path_integrator_NEE_DS_bw(map_object, vec3 x, vec3 w, float epsilon, vec3 dL_dR)
{
    // state for restarting

    SAVE_SEED(main_seed); // temporary seed to restore main sequence
    SAVE_SEED(before_path); // saved to restart main seed in the replay path

    vec3 ini_x = x;
    vec3 ini_w = w;

    float tMin, tMax;
    boundary(object, x, w, tMin, tMax);
    x += w * tMin;
    float d = tMax - tMin;

    // Accumulated radiance along the path
    vec3 R = vec3(0.0);
    vec3 R_after_sp = vec3(0.0);
    vec3 W = vec3(1.0);
    float null_W = 1.0; // factor to make the path-throughput robust

    float EPS = 0.00000001;
    bool scattered = false;

    while (d > EPS)
    {
        float md;
        float m = max(EPS, majorant(object, x, w, md) + 2*epsilon + 0.1); // extended majorant to support epsilon in the mininum and maximum cases
        float t = -log(1 - random()) / m; // Free flight
        x += w * min(t, md);
        if (t > md)
        {
            d -= md;
            continue;
        }
        if (t > d)
        break;

        float s = sigma(object, x);
        float Pc = (s + epsilon) / m;
        if (random() >= Pc) // null-collision
        {
            d -= t;
            W *= (1 - s/m)/(1 - Pc);
            continue;
        }

        scattered = true;

        if (s < EPS) // null sigma needs to be treated differently
        {
            if (null_W < EPS) // second chance of null sigma vanishes all path-throughput
            {
                W = vec3(0.0);
                break;
            }
            null_W = 0; // real Tr is 0 now but preserved in W the non-null part.
        }
        else
            W *= (s/m)/Pc;

        vec3 em = emission(object, x, -w);
        vec3 sa = scattering_albedo(object, x);
        vec3 added_R = W * (vec3(1.0) - sa) * em;

        // scattering factor
        W *= sa;

        // - Direct contribution
        vec3 wnee;
        vec3 env = environment_sampler(object, x, w, /*out*/ wnee);
        float T_nee = transmittance(object, x, wnee);
        float ph_nee = phase(object, x, w, wnee);
        added_R += W * T_nee * env * ph_nee;
        if (null_W < EPS)
            R_after_sp += added_R;
        else
            R += added_R;
        // - Indirect contribution
        vec3 wo;
        float rho = phase_sampler(object, x, w, wo);
        W *= rho;

        if (all(lessThan(W, vec3(EPS))))
        {
            W = vec3(0.0);
            break;
        }

        w = wo;
        boundary(object, x, w, tMin, d);
    }

    if (!scattered) // add direct radiance
        R += environment(object, w) * W;

    vec3 W_out = W;

    // restart states
    SET_SEED(before_path);
    x = ini_x;
    w = ini_w;

    // Accumulated radiance along the path
    W = vec3(1.0);

    // First eval
    boundary(object, x, w, tMin, tMax);
    x += w * tMin;
    d = tMax - tMin;

    while (d > EPS)
    {
        float md;
        float m = max(EPS, majorant(object, x, w, md) * 1.2 + 2 * epsilon);
        float t = -log(1 - random()) / m; // Free flight
        x += w * min(t, md);
        if (t > md)
        {
            d -= md;
            continue;
        }
        if (t > d) break;
        SAVE_SEED(before_sigma); // save seed for sigma bw
        float s = sigma(object, x);
        float Pc = (s + epsilon) / m;

        // R_out = (1 - s/m) * R_next + s/m * R_collision;
        // dL_dR_out = (1 - s/m) * dL_dRnext + s/m * dL_dRcol  => W *= (1 - s/m)/(1 - Pc) or ...
        // dL_ds = -1/m * R_next + 1/m * R_collision

        float dL_dsigma;
        if (random() >= Pc) // assume null collision
        {
            dL_dsigma = dot(dL_dR, -R / m / (1 - s/m));
            USING_SECONDARY_SEED(main_seed, before_sigma, sigma_bw(object, x, dL_dsigma));
            W *= (1 - s/m) / (1 - Pc); // needed to propagate gradient to next path positions
            d -= t;
            continue;
        }
        // Propagate sigma
        dL_dsigma = dot(dL_dR, s < EPS ? /* singular case */ R_after_sp / epsilon : R / s);
        USING_SECONDARY_SEED(main_seed, before_sigma, sigma_bw(object, x, dL_dsigma));

        W *= s / m / Pc; // update path-throughtput for gradient propagation

        if (s < EPS)
        return; // After singular vertex no more gradients are backprop

        SAVE_SEED(before_emission);
        vec3 em = emission(object, x, -w);
        SAVE_SEED(before_scattering_albedo);
        vec3 sa = scattering_albedo(object, x);
        vec3 dL_dem = dL_dR * W * (vec3(1.0) - sa);
        vec3 dL_dsa = dL_dR * W * (-1.0) * em;
        // Update R to represent only scattered Radiance
        R -= W * (vec3(1.0) - sa) * em;

        dL_dsa += dL_dR * R / max(vec3(EPS), sa);

        SAVE_SEED(before_environment_sampler);
        vec3 wnee;
        vec3 env = environment_sampler(object, x, w, /*out*/ wnee);
        SAVE_SEED(before_transmittance);
        float T_nee = transmittance(object, x, wnee);
        SAVE_SEED(before_phase);
        float ph_nee = phase(object, x, w, wnee);
        SAVE_SEED(before_phase_sampler);
        vec3 wo;
        float rho = phase_sampler(object, x, w, wo);
        SAVE_SEED(before_next_iteration);

        W *= sa;

        // R += W * T_nee * env * ph;
        float dL_dTnee = dot(dL_dR, W * env * ph_nee);
        vec3 dL_denv = dL_dR * W * T_nee * ph_nee;
        float dL_dphnee = dot(dL_dR, W * T_nee * env);

        R -= W * T_nee * env * ph_nee;

        float dL_drho = dot(dL_dR, R / rho);

        USING_SEED(before_emission, emission_bw(object, x, -w, dL_dem));
        USING_SEED(before_scattering_albedo, scattering_albedo_bw(object, x, dL_dsa));
        USING_SEED(before_environment_sampler, environment_sampler_bw(object, x, w, dL_denv));
        USING_SEED(before_transmittance, transmittance_bw(object, x, wnee, dL_dTnee));
        USING_SEED(before_phase, phase_bw(object, x, w, wnee, dL_dphnee));
        USING_SEED(before_phase_sampler, phase_sampler_bw(object, x, w, dL_drho));

        SET_SEED(before_next_iteration);

        W *= rho;

        if (all(lessThan(W, vec3(EPS))))
        return;

        w = wo;
        boundary(object, x, w, tMin, d);
    }
}



void accumulate_path_NEE_DT_fw(map_object, vec3 x, vec3 w, vec3 wnee, out vec3 A)
{
    A = vec3(0);
    vec3 Tr = vec3(1);
    float tMin, tMax;
    if (!boundary(object, x, w, tMin, tMax))
    return;
    float EPS = 0.00000001;

    float epsilon = parameters.ds_epsilon;

    x += w * tMin;
    float d = tMax - tMin;

    while (true) {

        float md;
        float m = max(EPS, majorant(object, x, w, md) + 2 * epsilon);
        float t = min(-log(1 - random()) / m, md); // Free flight
        x += w * min(t, md);
        if (t >= d - EPS)
        return;
        if (t >= md - EPS)
        {
            d -= md;
            continue;
        }

        float s = sigma(object, x) + EPS;
        float Pc = (s + epsilon) / m;
        if (random() >= Pc) // null collision
        {
            d -= t;
            Tr *= (1 - s/m) / (1 - Pc);
            continue;
        }

        Tr *= s / m / Pc;

        Tr *= scattering_albedo(object, x);

        // sample NEE direction
        float T_nee = transmittance(object, x, wnee);
        A += Tr * T_nee * phase(object, x, w, wnee);

//        if (s < EPS)
//        return; // After singular vertex, no propagate bias to next scatters

        // continue path
        vec3 wo;
        float rho = phase_sampler(object, x, w, wo);
        Tr *= rho;
        w = wo;
        boundary (object, x, w, tMin, tMax);
        x += w*tMin;
        d = tMax - tMin;
    }
}

void accumulate_path_NEE_DT_bw(map_object, vec3 x, vec3 w, vec3 wnee, vec3 A, vec3 dL_dA)
{
    vec3 Tr = vec3(1);
    float tMin, tMax;
    if (!boundary(object, x, w, tMin, tMax))
    return;

    SAVE_SEED(main_seed);

    float EPS = 0.00000001;

    float epsilon = parameters.ds_epsilon;

    x += w * tMin;
    float d = tMax - tMin;

    while (true) {

        float md;
        float m = max(EPS, majorant(object, x, w, md) + 2 * epsilon);
        float t = min(-log(1 - random()) / m, md); // Free flight
        x += w * min(t, md);
        if (t >= d - EPS)
        return;
        if (t >= md - EPS)
        {
            d -= md;
            continue;
        }

        SAVE_SEED(before_sigma);
        float s = sigma(object, x) + EPS;
        float Pc = (s + epsilon) / m;
        if (random() >= Pc) // null collision
        {
            float dL_dsigma = dot(dL_dA, -A / m / (1 - s/m));
            USING_SECONDARY_SEED(main_seed, before_sigma, sigma_bw(object, x, dL_dsigma));

            d -= t;
            Tr *= (1 - s/m) / (1 - Pc);
            continue;
        }

        {
            float dL_dsigma = dot(dL_dA, A / s);
            USING_SECONDARY_SEED(main_seed, before_sigma, sigma_bw(object, x, dL_dsigma));
            Tr *= s / m / Pc;
        }

        Tr *= scattering_albedo(object, x);

        // sample NEE direction
        SAVE_SEED(before_Tnee);
        float T_nee = transmittance(object, x, wnee);
        float ph = phase(object, x, w, wnee);

        float dL_dTnee = dot(dL_dA, Tr * ph);
        USING_SECONDARY_SEED(main_seed, before_Tnee, transmittance_bw(object, x, wnee, dL_dTnee));
        A -= Tr * T_nee * ph;

        if (s < EPS)
        return; // After singular vertex, no propagate bias to next scatters

        // continue path
        vec3 wo;
        float rho = phase_sampler(object, x, w, wo);
        Tr *= rho;
        w = wo;
        boundary (object, x, w, tMin, tMax);
        x += w*tMin;
        d = tMax - tMin;
    }
}


