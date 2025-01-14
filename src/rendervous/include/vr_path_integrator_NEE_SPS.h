bool sample_collision_position(map_object,
    vec3 x, vec3 w, float d,
    out vec3 xc, // sampled point
    out float weight_xc, // weight of the sampling  ~T or ~Tsigma if singular
    out float sigma_xc, // sigma at collision point
    out int samples,
    inout bool mis_allowed // indicates if MIS is used or not, out as false if transmittance was used
    ){
    float MIS_BETA = 3;
    float t = 0;
    weight_xc = 0.0;
    sigma_xc = 0.0;
    samples = 0;

    while (true) {
        float md;
        float m = majorant(object, xc, w, md) + 0.000001; // to avoid creating outliers with s / m
        float dt = min(-log(1 - random()) / m, min(md, d - t));
        t += dt;
        xc = w * t + x;
        if (dt >= md - 0.0000001)
        continue; // free flight in majorant subdomain
        if (t >= d - 0.0000001)
        break; // endpoint reached
        float s = sigma(object, xc);
        samples++;
        float Ps = s / m;
        if (random() < Ps) // real collision
        {
            weight_xc = 1.0;
            sigma_xc = s;
            break;
        }
    }

    if (!mis_allowed)
    return false;

    float powered_c = pow(sigma_xc, MIS_BETA);

    // mis weight for the collision (weighted again wrt sampler)
    float mis_wc = weight_xc * powered_c / (1 + powered_c);

    float weight_f = t; // normalized pdf
    //float sample_f = (1 - random()) * t;
    //vec3 xf = x + w * sample_f;
    vec3 xf = mix(xc, x, random());
    float sigma_f = sigma(object, xf);
    float powered_f = pow(sigma_f, MIS_BETA);
    float mis_wf = t / (1 + powered_f);

    weight_xc = (mis_wc + mis_wf);
    if (random() * weight_xc < mis_wf) // bounce in potential zero region
    {
        //weight_xc *= weight_f;
        xc = xf;
        sigma_xc = sigma_f;
        mis_allowed = false; // mark the singular vertex was already hit
        return true; // singular position was hit
    }

    return false; // was no singular hit, check weight_c to know if hit at all.
}


//void st_tau_bw(map_object, vec3 a, vec3 b, int bins, float dL_dtau)
//{
//    if (dL_dtau == 0)
//    return;
//
//    float d = length(a - b);
//    float bin_size = d / bins;
//    float dL_dsigma = dL_dtau * bin_size;
//    for (int s = 0; s < bins; s ++)
//    {
//        vec3 xt = mix(b, a, (s + random()) / bins);
//        sigma_bw(object, xt, dL_dsigma);
//    }
//}

void st_tau_bw(map_object, vec3 a, vec3 b, float dL_dtau)
{
    if (dL_dtau == 0)
    return;

    float d = length(a - b);
    float dL_dsigma = dL_dtau * d;
    sigma_bw(object, mix(a,b, random()), dL_dsigma);
}


void st_tau_bw(map_object, vec3 a, vec3 b, int bins, float dL_dtau)
{
    if (dL_dtau == 0)
    return;

    if (bins == 0)
    return;

    float d = length(a - b);
    float dL_dsigma = dL_dtau * d / bins;

    for (int i=0; i<bins; i++)
        sigma_bw(object, mix(b, a, (i+random())/bins), dL_dsigma);
}


void path_integrator_NEE_SPS_bw(map_object, vec3 x, vec3 w, vec3 dL_dR)
{
    float tMin, tMax;
    boundary(object, x, w, tMin, tMax);
    x += w*tMin;
    float d = tMax - tMin;

    if (d <= 0)
    return;

    BRANCH_SEED(secondary_seed); // branched sequence to update tau gradients between scatters.
    SAVE_SEED(main_seed); // temporal seed to save main seed while secondary is used

    // Consider radiance seen direct through
    float direct_T = transmittance(object, x, w); // No scatter case handled appart
    vec3 direct_environment = environment(object, w);
    // R_direct = direct_T * direct_environment;
    vec3 dL_ddirectenv = dL_dR * direct_T;
    float dL_ddirectT = dot(dL_dR, direct_environment);
    SWITCH_SEED(main_seed, secondary_seed);
    transmittance_bw(object, x, w, dL_ddirectT);
    environment_bw(object, w, dL_ddirectenv);
    SWITCH_SEED(secondary_seed, main_seed);

    // Accumulated radiance along the primary path
    vec3 R = vec3(0); // NEE accumulation before the singular vertex
    vec3 Rf = vec3(0); // NEE accumulation from singular vertex, without singular factor
    vec3 W = vec3(1); // Path throughput without factor
    float W_singular = 1.0; // Path throughput factor at singular vertex

    bool mis_allowed = true;
    vec3 ini_x = x;
    vec3 ini_w = w;
    float ini_d = d;

    //float EPS = 0.00000001;
    float EPS = 0.00001;

    SAVE_SEED(before_path);

    while (true) {
        // Sample Collision and Transmittance
        float weight_xc;
        vec3 xc; float sigma_xc;
        int samples;
        if (sample_collision_position( // Sample a collision with MIS between DT and DRT. Returns if sample was possible. If mis_allowed if false, DT is used always.
                object,
                x, w, d,
                xc, weight_xc, sigma_xc, samples, mis_allowed))
                W_singular = sigma_xc; // singular vertex

        if (weight_xc == 0.0) // no collision
        break;

        x = xc;
        W *= weight_xc * scattering_albedo(object, x);

        // sample NEE direction
        vec3 wnee;
        vec3 nee_env = environment_sampler(object, x, w, wnee);
        float nee_T = transmittance(object, x, wnee);
        float nee_rho = phase(object, x, w, wnee);
        vec3 added_R = W * nee_T * nee_rho * nee_env;
        R += int(mis_allowed) * added_R;
        Rf += int(!mis_allowed) * added_R;

        // continue path
        vec3 wo;
        float rho = phase_sampler(object, x, w, wo);
        W *= rho;
        w = wo;
        boundary(object, x, w, tMin, d);

        if (all(lessThan(W, vec3(EPS))))
        break;
    }

    W = vec3(1);
    mis_allowed = true;
    x = ini_x;
    w = ini_w;
    d = ini_d;
    vec3 prev_x = x;

    SET_SEED(before_path); // Replay!

    while (true) {
        // Sample Collision and Transmittance
        float weight_xc;
        vec3 xc; float sigma_xc;
        int samples;
        bool singular_vertex = sample_collision_position( // Sample a collision with MIS between DT and DRT. Returns if sample was possible. If mis_allowed if false, DT is used always.
                object,
                x, w, d,
                xc, weight_xc, sigma_xc, samples, mis_allowed);

        if (weight_xc == 0.0) // no collision
        break;

        vec3 further_R = R + W_singular * Rf;
        float k = dot(dL_dR, further_R);
        float dL_dtau = -k;
        float dL_dsigma = singular_vertex ? dot(dL_dR, Rf) : k / max(0.001, sigma_xc); // to avoid outliers
        // R = <T(x, xc)> * further_R

        x = xc;
        W *= weight_xc;

        vec3 sa = scattering_albedo(object, x);
        /* TODO: Consider also emission here */

        W *= sa;

        vec3 wnee;
        vec3 env = environment_sampler(object, x, w, /*out*/ wnee);
        float T_nee = transmittance(object, x, wnee);
        float ph_nee = phase(object, x, w, wnee);
        vec3 wo;
        float rho = phase_sampler(object, x, w, wo);

        // R += W * T_nee * env * ph;
        float dL_dTnee = dot(dL_dR, W * (mis_allowed ? 1.0 : W_singular) * env * ph_nee);
        //vec3 dL_denv = dL_dR * W * T_nee * ph_nee;
        //float dL_dphnee = dot(dL_dR, W * T_nee * env);


        vec3 added_R = W * T_nee * ph_nee * env;

        R -= int(mis_allowed)*added_R;
        Rf -= int(!mis_allowed)*added_R;

        //float dL_drho = dot(dL_dR, R / rho);
        SWITCH_SEED(main_seed, secondary_seed);
        sigma_bw(object, x, dL_dsigma);
        st_tau_bw(object, x, prev_x, max(samples, 1), dL_dtau);
//        st_tau_bw(object, x, prev_x, max(samples, 1), dL_dtau);

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

        if (all(lessThan(W, vec3(EPS))))
        break;

        prev_x = x;
    }
}

