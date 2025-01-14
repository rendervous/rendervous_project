/*
x: ray position (assumed inside volume)
w: ray direction
d: distance of path inside volume.
out t: selected position x + wt
out weight_t: selected position weight, should be estimator <T(t)s(t)>/s(t) divided by p(t)
*/
void sample_collision_position_ds(map_object, vec3 x, vec3 w, float d, float epsilon, out float t, out float weight_t)
{
    float EPS = 0.0000001;
    t = 0;
    weight_t = 1.0;
    while (true) {
        float md;
        float m = max(EPS, majorant(object, x, w, md) + 2 * epsilon + 0.1);
        float dt = min(-log(1 - random()) / m, min(md, d - t));

        t += dt;

        if (dt >= md - EPS)
            continue;

        if (t >= d - EPS)
        break; // endpoint reached

        vec3 xt = x + w * t;
        float s = sigma(object, x + w * t);
        float Pc = (s + epsilon) / m;
        if (random() < Pc)
        {
            weight_t *= 1.0/(s + epsilon); // (s/m/Pc)/s;
            return;
        }
        weight_t *= (1 - s/m)/(1 - Pc);
    }

    weight_t = 0; // no collision
}

void path_integrator_NEE_DRTDS_bw(map_object, vec3 x, vec3 w, float epsilon, vec3 dL_dR)
{
    float tMin, tMax;
    boundary(object, x, w, tMin, tMax);
    x += w * tMin;
    float d = tMax - tMin;

    if (d <= 0)
    return;

    BRANCH_SEED(secondary_seed);
    SAVE_SEED(main_seed); // temporary seed to restore main sequence

    // Consider radiance seen direct through
    float direct_T = transmittance(object, x, w); // No scatter case handled appart
    vec3 direct_environment = environment(object, w);
    // R_direct = direct_T * direct_environment;
    vec3 dL_ddirectenv = dL_dR * direct_T;
    float dL_ddirectT = dot(dL_dR, direct_environment);
    SWITCH_SEED(main_seed, secondary_seed);
    environment_bw(object, w, dL_ddirectenv);
    transmittance_bw(object, x, w, dL_ddirectT);
    SWITCH_SEED(secondary_seed, main_seed);

    // Accumulated radiance along the primary path
    vec3 R = vec3(0.0);
    vec3 W = vec3(1.0);

    vec3 ini_x = x;
    vec3 ini_w = w;
    float ini_d = d;

    // resevoir ray
    vec3 res_x = x;
    vec3 res_w = w;
    vec3 res_W = W;
    float res_P = 3; // W.x + W.y + W.z
    float total_weight = 3;

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

    SET_SEED(before_path);

    while (d > EPS)
    {
        float md;
        float m = majorant(object, x, w, md)*1.05 + 0.1;
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

//        vec3 em = emission(object, x, -w);
        vec3 sa = scattering_albedo(object, x);
//        vec3 dL_dem = dL_dR * W * (vec3(1.0) - sa);
//        vec3 dL_dsa = dL_dR * W * (-1.0) * em;
        // Update R to represent only scattered Radiance
//        R -= W * (vec3(1.0) - sa) * em;

//        dL_dsa += dL_dR * R / max(vec3(EPS), sa);

        vec3 wnee;
        vec3 env = environment_sampler(object, x, w, /*out*/ wnee);
        float T_nee = transmittance(object, x, wnee);
        float ph_nee = phase(object, x, w, wnee);
        vec3 wo;
        float rho = phase_sampler(object, x, w, wo);

        W *= sa;

        // R += W * T_nee * env * ph;
        float dL_dTnee = dot(dL_dR, W * env * ph_nee);
//        vec3 dL_denv = dL_dR * W * T_nee * ph_nee;
//        float dL_dphnee = dot(dL_dR, W * T_nee * env);

        R -= W * T_nee * env * ph_nee;

//        float dL_drho = dot(dL_dR, R / rho);
        SWITCH_SEED(main_seed, secondary_seed);
//        USING_SEED(before_emission, emission_bw(object, x, -w, dL_dem));
//        USING_SEED(before_scattering_albedo, scattering_albedo_bw(object, x, dL_dsa));
//        USING_SEED(before_environment_sampler, environment_sampler_bw(object, x, w, dL_denv));
        transmittance_bw(object, x, wnee, dL_dTnee);
//        USING_SEED(before_phase, phase_bw(object, x, w, wnee, dL_dphnee));
//        USING_SEED(before_phase_sampler, phase_sampler_bw(object, x, w, dL_drho));
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
    sample_collision_position_ds(object, res_x, res_w, d, epsilon, s_t, weight_t);
    if (weight_t > 0){
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

        // res_W: resevoir path-throughput till xs
        // weight_t: <T>/pdf(t)
        // rho_s: <rho>/pdf(ws)
        float dL_dsigmascattering = dot(dL_dR, Ls) * total_weight / res_P;
        sigma_bw(object, xs, dL_dsigmascattering);
    }
}