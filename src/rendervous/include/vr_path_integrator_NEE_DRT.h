/*
x: ray position (assumed inside volume)
w: ray direction
d: distance of path inside volume.
out t: selected position x + wt
out weight_t: selected position weight, should be estimator <T(t)> divided by p(t)
*/
float sample_transmittance_position(map_object, vec3 x, vec3 w, float d, out float sample_t, out float weight_t)
{
    float T = 1.0;
    float t = 0;
    float res_t = 0;
    float res_dt = 0;
    float C = 0;
    float EPS = 0.0000001;
    while (true) {
        float md;
        float m = max(EPS, majorant(object, x, w, md));
        float dt = min(-log(1 - random()) / m, min(md, d - t));

        float w_step = T * dt;
        C += w_step;
        if (random() < w_step / C)
        {
            res_t = t;
            res_dt = dt;
        }

        t += dt;

        if (dt >= md - EPS)
            continue;

        if (t >= d - EPS)
        break; // endpoint reached

        vec3 xt = x + w * t;
        T *= 1 - sigma(object, x + w * t) / m;
    }

    weight_t = C; // normalized pdf
    sample_t = res_t + random() * res_dt;
    return T; // Total transmittance
}

void path_integrator_NEE_DRT_bw(map_object, vec3 x, vec3 w, vec3 dL_dR)
{
    float tMin, tMax;
    boundary(object, x, w, tMin, tMax);
    x += w * tMin;
    float d = tMax - tMin;

    if (d <= 0)
    return;

    // state for restarting and run secondary seeds
    BRANCH_SEED(secondary_seed);
    SAVE_SEED(main_seed); // temporary seed to restore main sequence

    // Consider radiance seen direct through
    float direct_T = transmittance(object, x, w);
    vec3 direct_env = environment(object, w);
    // return W*environment(object, w) + R;
    vec3 dL_denv = dL_dR * direct_T;
    float dL_ddT = dot(dL_dR, direct_env);
    SWITCH_SEED(main_seed, secondary_seed);
    transmittance_bw(object, x, w, dL_ddT);
    environment_bw(object, w, dL_denv);
    SWITCH_SEED(secondary_seed, main_seed);

    // Accumulated radiance along the primary path
    vec3 R = vec3(0.0);
    vec3 W = vec3(1.0);

    // resevoir ray
    vec3 res_x = x;
    vec3 res_w = w;
    vec3 res_W = W;
    float res_P = 3; // W.x + W.y + W.z
    float total_weight = 3;

    vec3 ini_x = x;
    vec3 ini_w = w;
    float ini_d = d;

    float EPS = 0.00000001;

    SAVE_SEED(before_path); // saved to restart main seed in the replay path

    while (d > EPS)
    {
        float md;
        float m = majorant(object, x, w, md)*1.05 + 0.1; // extended majorant to support epsilon in the maximum cases
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
        float Pc = s / m;
        if (random() >= Pc) // null-collision
        {
            d -= t;
            continue;
        }

//        vec3 em = emission(object, x, -w);
        vec3 sa = scattering_albedo(object, x);
//        vec3 added_R = W * (vec3(1.0) - sa) * em;

        // scattering factor
        W *= sa;

        // - Direct contribution
        vec3 wnee;
        vec3 env = environment_sampler(object, x, w, /*out*/ wnee);
        float T_nee = transmittance(object, x, wnee);
        float ph_nee = phase(object, x, w, wnee);
        vec3 added_R = W * T_nee * env * ph_nee;
        R += added_R;

        // - Indirect contribution
        vec3 wo;
        float rho = phase_sampler(object, x, w, wo);
        W *= rho;
        w = wo;
        boundary(object, x, w, tMin, d);

        // Update the resevoir
        float current_P = W.x + W.y + W.z;
        total_weight += current_P;
        if (random() < current_P / total_weight) {
            res_P = current_P;
            res_W  = W;
            res_x = x;
            res_w = w;
        }

        if (all(lessThan(W, vec3(EPS))))
            break;
    }

    // Replay but updating scattering only with the MIS weight of real collision,
    // The gradient wrt to Transmittance in the resevoir point will be added afterwards.

    // restart states
    W = vec3(1.0);
    x = ini_x;
    w = ini_w;
    d = ini_d;

    // Replay
    SET_SEED(before_path);

    while (d > EPS)
    {
        float md;
        float m = majorant(object, x, w, md)*1.05 + 0.1; // extended majorant to support epsilon in the maximum cases
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
        float Pc = s / m;

        // R_out = (1 - s/m) * R_next + s/m * R_collision;
        // dL_dR_out = (1 - s/m) * dL_dRnext + s/m * dL_dRcol  => W *= (1 - s/m)/(1 - Pc) or ...
        // dL_ds = -1/m * R_next + 1/m * R_collision

        float dL_dsigma;
        if (random() >= Pc) // assume null collision
        {
            dL_dsigma = dot(dL_dR, -R / m / (1 - s/m));
            SWITCH_SEED(main_seed, secondary_seed);
            sigma_bw(object, x, dL_dsigma);
            SWITCH_SEED(secondary_seed, main_seed);
            d -= t;
            continue;
        }

        float mis_weight_Ts = s*s / (1 + s*s);
        // Propagate sigma
        dL_dsigma = dot(dL_dR, mis_weight_Ts * R / s);

//        vec3 em = emission(object, x, -w);
        vec3 sa = scattering_albedo(object, x);
//        vec3 dL_dem = dL_dR * W * (vec3(1.0) - sa);
//        vec3 dL_dsa = dL_dR * W * (-1.0) * em;
        // Update R to represent only scattered Radiance
//        R -= W * (vec3(1.0) - sa) * em;

//        dL_dsa += dL_dR * R / max(vec3(EPS), sa);

        W *= sa;

        vec3 wnee;
        vec3 env = environment_sampler(object, x, w, /*out*/ wnee);
        float T_nee = transmittance(object, x, wnee);
        float ph_nee = phase(object, x, w, wnee);
        vec3 wo;
        float rho = phase_sampler(object, x, w, wo);

        // R += W * T_nee * env * ph;
        float dL_dTnee = dot(dL_dR, W * env * ph_nee);
//        vec3 dL_denv = dL_dR * W * T_nee * ph_nee;
//        float dL_dphnee = dot(dL_dR, W * T_nee * env);

        R -= W * T_nee * env * ph_nee;

//        float dL_drho = dot(dL_dR, R / rho);

        SWITCH_SEED(main_seed, secondary_seed);
        sigma_bw(object, x, dL_dsigma);
        //USING_SEED(before_emission, emission_bw(object, x, -w, dL_dem));
        //USING_SEED(before_scattering_albedo, scattering_albedo_bw(object, x, dL_dsa));
        //USING_SEED(before_environment_sampler, environment_sampler_bw(object, x, w, dL_denv));
        transmittance_bw(object, x, wnee, dL_dTnee);
        //USING_SEED(before_phase, phase_bw(object, x, w, wnee, dL_dphnee));
        //USING_SEED(before_phase_sampler, phase_sampler_bw(object, x, w, dL_drho));
        SWITCH_SEED(secondary_seed, main_seed);

        W *= rho;

        w = wo;
        boundary(object, x, w, tMin, d);

        advance_random(); // to match random used by resevoir sampling

        if (all(lessThan(W, vec3(EPS))))
        break;
    }

    // ** Add scattering sigma gradient at reseivoir vertex
    // * test against boundary
    boundary(object, res_x, res_w, tMin, d);
    // Sample a transmittance based position for the scattering event
    float s_t, weight_t;
    sample_transmittance_position(object, res_x, res_w, d, s_t, weight_t);
    // scattering position to propagate gradient
    vec3 xs = res_x + res_w * s_t;
    // Direct radiance at scattered position (important to reduce variance)
    vec3 ls_direct_w;
    vec3 Ls_direct_env = environment_sampler(object, xs, res_w, ls_direct_w);
    vec3 Ls = transmittance(object, xs, ls_direct_w) * Ls_direct_env * phase(object, xs, res_w, ls_direct_w);
    vec3 ws ;
    float rho_s = phase_sampler(object, xs, res_w, ws);
    Ls += rho_s * path_integrator_NEE(object, xs, ws, true); // true-only scattering contribution
    Ls *= res_W * weight_t * scattering_albedo(object, xs);

    float sigma_xs = sigma(object, xs); // necessary for MIS weight
    // res_W: resevoir path-throughput till xs
    // weight_t: <T>/pdf(t)
    // rho_s: <rho>/pdf(ws)
    float mis_weight_T = 1 / (1 + sigma_xs * sigma_xs);
    float dL_dsigmascattering = dot(dL_dR, Ls) * mis_weight_T * total_weight / res_P;
    SWITCH_SEED(main_seed, secondary_seed);
    sigma_bw(object, xs, dL_dsigmascattering);
    SWITCH_SEED(secondary_seed, main_seed);
}