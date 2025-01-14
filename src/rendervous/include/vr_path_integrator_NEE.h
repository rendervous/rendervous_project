vec3 path_integrator_NEE(map_object, vec3 x, vec3 w, bool only_scatters)
{
    float tMin, tMax;
    boundary(object, x, w, tMin, tMax);
    x += w * tMin;
    float d = tMax - tMin;

    // Accumulated radiance along the path
    vec3 R = vec3(0.0);
    vec3 W = vec3(1.0);

    // transmitted radiance
    if (!only_scatters)
        R += transmittance(object, x, w) * environment(object, w);

    float EPS = 0.00001;

    while (d > EPS)
    {
        float md;
        float m = max(EPS, majorant(object, x, w, md));
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

        // Add emissive term
        R += W * (vec3(1.0) - sa) * em;

        // Add scattering term
        W *= sa;

        // - Direct contribution
        vec3 wnee;
        vec3 env = environment_sampler(object, x, w, /*out*/ wnee);
        float T_nee = transmittance(object, x, wnee);
        R += W * T_nee * env * phase(object, x, w, wnee);
        // - Indirect contribution
        vec3 wo;
        float rho = phase_sampler(object, x, w, wo);
        W *= rho;

        if (all(lessThan(W, vec3(EPS))))
        return R;

        w = wo;
        boundary(object, x, w, tMin, d);
    }

    return R;
}
