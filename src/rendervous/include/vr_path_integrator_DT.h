/*
Delta-tracking path integrator. High-variance but good as ground-truth
requires: boundary, sigma, scattering_albedo, emission, phase_sampler, environment
bw requires: ds_epsilon
*/

vec3 path_integrator_DT(map_object, vec3 x, vec3 w)
{
    float tMin, tMax;
    boundary(object, x, w, tMin, tMax);
    x += w * tMin;
    float d = tMax - tMin;

    // Accumulated radiance along the path
    vec3 R = vec3(0.0);
    vec3 W = vec3(1.0);

    while (d > 0.00000001)
    {
        float md;
        float m = max(0.00000001, majorant(object, x, w, md));
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

        vec3 sa = scattering_albedo(object, x);
        vec3 em = emission(object, x, -w);

        R += W * (vec3(1.0) - sa) * em;
        W *= sa;

        vec3 wo;
        float rho = phase_sampler(object, x, w, wo);
        W *= rho;

        if (all(lessThan(W, vec3(0.0000001))))
        return R;

        w = wo;
        boundary(object, x, w, tMin, d);
    }

    return W*environment(object, w) + R;
}

void path_integrator_DT_bw(map_object, vec3 x, vec3 w, float epsilon, vec3 dL_dR)
{
    // state for restarting
    uvec4 seed_before_path = get_seed();
    vec3 ini_x = x;
    vec3 ini_w = w;

    float tMin, tMax;
    boundary(object, x, w, tMin, tMax);
    x += w * tMin;
    float d = tMax - tMin;
    float EPS = 0.00000001;

    // Accumulated radiance along the path
    //vec3 R = vec3(0.0);
    //vec3 R_after_sp = vec3(0.0);
    vec3 W = vec3(1.0);
    float null_W = 1.0; // factor to make the path-throughput robust

    while (d > EPS)
    {
        float md;
        float m = max(EPS, majorant(object, x, w, md) + 2 * epsilon); // extended majorant to support epsilon in the mininum and maximum cases
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

        if (s == 0.0) // null sigma needs to be treated differently
        {
            if (null_W == 0) // second chance of null sigma vanishes all path-throughput
            {
                W = vec3(0.0);
                break;
            }
            null_W = 0; // real Tr is 0 now but preserved in W the non-null part.
        }
        else
            W *= (s/m)/Pc;

        //vec3 em = emission(object, x, -w);
        vec3 sa = scattering_albedo(object, x);
//        vec3 added_R = W * (vec3(1.0) - sa) * em;
//        if (null_W == 1.0)
//            R += added_R;
//        else
//            R_after_sp += added_R;

        vec3 wo;
        float rho = phase_sampler(object, x, w, wo);
        W *= sa * rho;

        if (all(lessThan(W, vec3(EPS))))
        {
            W = vec3(0.0);
            break;
        }

        w = wo;
        boundary(object, x, w, tMin, d);
    }

    vec3 env = environment(object, w);

//    if (null_W == 1.0)
//        R += W * env;
//    else
//        R_after_sp += W * env;

    vec3 dL_dW = dL_dR * env;

    // return W*environment(object, w) + R;
//    vec3 dL_denv = dL_dR * W * null_W;
//    environment_bw(object, w, dL_denv);

    // restart states
    set_seed(seed_before_path);
    x = ini_x;
    w = ini_w;

    // First eval
    boundary(object, x, w, tMin, tMax);
    x += w * tMin;
    d = tMax - tMin;

    // Accumulated radiance along the path
    vec3 W_out = W;

    W = vec3(1.0);

    while (d > EPS)
    {
        float md;
        float m = max(EPS, majorant(object, x, w, md) + 2*epsilon);
        float t = -log(1 - random()) / m; // Free flight
        x += w * min(t, md);
        if (t > md)
        {
            d -= md;
            continue;
        }
        if (t > d) break;
        float s = sigma(object, x);
        float Pc = (s + epsilon) / m;

        // R_out = (1 - s/m) * R_next + s/m * R_collision;
        // dL_dR_out = (1 - s/m) * dL_dRnext + s/m * dL_dRcol  => W *= (1 - s/m)/(1 - Pc) or ...
        // dL_ds = -1/m * R_next + 1/m * R_collision

        if (random() >= Pc) // assume null collision
        {
//            float dL_dsigma = dot(dL_dR, vec3(1.0));
            float dL_dsigma = dot(dL_dW, -W_out * null_W / m / (1 - s/m));
            sigma_bw(object, x, dL_dsigma);
            W *= (1 - s/m) / (1 - Pc); // needed to propagate gradient to next path positions
            d -= t;
            continue;
        }
        // Propagate sigma
        {
            float dL_dsigma = dot(dL_dW, s == 0 ? /* singular case */ W_out : W_out * null_W / s);
            sigma_bw(object, x, dL_dsigma);
        }

        W *= s / m / Pc; // update path-throughtput for gradient propagation

        if (s == 0.0)
        break; // After singular vertex no more gradients are backprop

//        vec3 em = emission(object, x, -w);
        vec3 sa = scattering_albedo(object, x);
        // Propagate emission
        {
//            vec3 dL_dem = dL_dR * W * (vec3(1.0) - sa);
            //emission_bw(object, x, -w, dL_dem);
//            vec3 dL_dsa = dL_dR * W * (-1.0) * em;
            //scattering_albedo_bw(object, x, dL_dsa);
        }
        // Update R to represent only scattered Radiance
//        R -= W * (vec3(1.0) - sa) * em;
        // Propagate scattering
        {
//            vec3 dL_dalbedo = dL_dR * R / max(vec3(EPS), sa);
            //scattering_albedo_bw(object, x, dL_dalbedo);
        }
        // Propagate phase
        vec3 wo;
        float rho = phase_sampler(object, x, w, wo);
        {
//            float dL_drho = dot(dL_dR, R / rho);
            //phase_sampler_bw(object, x, w, dL_drho);
        }

        W *= sa;
        W *= rho;

        if (all(lessThan(W, vec3(EPS))))
        return;

        w = wo;
        boundary(object, x, w, tMin, d);
    }
}


//vec3 DT_SamplePath_DS(
//    vec3 x, vec3 w, out float Tr_nullifier) {
//    vec3 Tr = vec3(1);
//    Tr_nullifier = 1.0;
//    float tMin, tMax;
//    if (!intersect_ray_box(x, w, b_min, b_max, tMin, tMax))
//        return Tr;
//    x += w*tMin;
//    float d = tMax - tMin;
//    float MAJORANT = SIGMA_SCALE * (1.05 + epsilon);
//    while (true){
//        float dt = min(-log(1 - random(seed)) / MAJORANT, d);
//        x += dt * w;
//        if (dt >= d - 0.000001)
//            break;
//        float sigma = sample_sigma(x) * SIGMA_SCALE;
//        float Ps = (sigma + SIGMA_SCALE * epsilon) / MAJORANT;
//        if (random(seed) < 1 - Ps) // null collision
//        {
//            Tr *= (1 - sigma / MAJORANT) / (1 - Ps);
//            d -= dt;
//            continue;
//        }
//        Tr *= sample_albedo(x);
//        if (sigma <= 0.0)
//        {
//            if (Tr_nullifier == 0)
//            return vec3(0); // 2nd zero bounce -> null path even for gradients
//            Tr_nullifier = 0; // real Tr is 0 now
//        }
//        else
//            Tr *= sigma / MAJORANT / Ps;
//
//        w = SampleHG (seed, w, PHASE_G);
//        intersect_ray_box(x, w, b_min, b_max, tMin, tMax);
//        x += w*tMin;
//        d = tMax - tMin;
//    }
//
//    return Tr;
//}
//
//
//void DT_SamplePath_DS_bw(inout uvec4 seed, vec3 x, vec3 w, float epsilon, vec3 b_min, vec3 b_max, vec3 Tr_out, float Tr_nullifier_out, vec3 dL_dTr)
//{
//    if (all(equal(Tr_out, vec3(0, 0, 0))))
//    return;
//
//    float tMin, tMax;
//    if (!intersect_ray_box(x, w, b_min, b_max, tMin, tMax))
//        return;
//    x += w * tMin;
//    float d = tMax - tMin;
//    vec3 Tr = vec3(1);
//    float MAJORANT = SIGMA_SCALE * (1.05 + epsilon);
//
//    while (true){
//        float dt = min(-log(1 - random(seed)) / MAJORANT, d);
//        x += dt * w;
//        if (dt >= d - 0.000001)
//            break;
//        float sigma = sample_sigma(x) * SIGMA_SCALE;
//        float Ps = (sigma + SIGMA_SCALE * epsilon) / MAJORANT;
//
//        if (random(seed) < 1 - Ps) // null collision
//        {
//            d -= dt;
//            // Backprop through null collision part
//            // L_out = (1 - voxel_density) * L_in / (1 - Ps) + (voxel_density) * (...)
//            // dL_dPn = L_in
//            vec3 dTr_dPn = Tr_out * Tr_nullifier_out / (1 - sigma / MAJORANT);
//            float dPn_dsigma = -SIGMA_SCALE / MAJORANT;
//            float dL_dsigma = dot(dL_dTr * dTr_dPn * dPn_dsigma, vec3(1,1,1));
//            sample_sigma_bw(x, dL_dsigma);
//            continue;
//        }
//        else
//        {
//            // Backprop through extinction collision part
//            // L_out = (1 - voxel_density) * L_in / (1 - Ps) + voxel_density * Ls * scattering_albedo / Ps
//            // dL_dPt = L_in * scattering_albedo
//            // Tr_out * nullifier / (sigma / majorant / Ps) / Ps
//            vec3 dTr_dPt = (sigma <= 0.0 ? Tr_out / Ps : Tr_out * Tr_nullifier_out * MAJORANT / sigma);
//            float dPt_dsigma = SIGMA_SCALE / MAJORANT;
//            float dL_dsigma = dot(dL_dTr * dTr_dPt * dPt_dsigma, vec3(1,1,1));
//            sample_sigma_bw(x, dL_dsigma);
//
//            vec3 albedo = sample_albedo(x);
//            vec3 dTr_dAlbedo = Tr_out / albedo;
//            vec3 dL_dalbedo = dL_dTr * dTr_dAlbedo;
//            sample_albedo_bw(x, dL_dalbedo);
//            Tr *= albedo;
//        }
//
//        w = SampleHG (seed, w, PHASE_G);
//        float tMin, tMax;
//        intersect_ray_box(x, w, b_min, b_max, tMin, tMax);
//        x += w * tMin;
//        d = tMax - tMin;
//    }
//}
