/*
Delta-tracking path sampler. Gives the final direction of the path and the importance
requires: boundary, sigma, scattering_albedo, phase_sampler
*/

void path_sampler_DT(map_object, vec3 x, vec3 wini, out vec3 w, out vec3 W)
{
    w = wini;
    W = vec3(1.0);
    float tMin, tMax;
    boundary(object, x, w, tMin, tMax);
    x += w * tMin;
    float d = tMax - tMin;
    // Importance along the path

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
        return; // leaves the volume

        float s = sigma(object, x);
        float Pc = s / m;
        if (random() >= Pc) // null-collision
        {
            d -= t;
            continue;
        }

        W *= scattering_albedo(object, x);

        vec3 wo;
        float rho = phase_sampler(object, x, w, wo);
        W *= rho;

        if (all(lessThan(W, vec3(0.0000001))))
        {
            W = vec3(0.0); // fully absorbed photon
            return;
        }

        w = wo;
        boundary(object, x, w, tMin, d);
    }
}


void path_sampler_DT_bw(map_object, vec3 x, vec3 w, vec3 dL_dW)
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

    vec3 W = vec3(1.0);

    while (d > EPS)
    {
        float md;
        float m = max(EPS, majorant(object, x, w, md)); // extended majorant to support epsilon in the mininum and maximum cases
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
        float m = max(EPS, majorant(object, x, w, md));
        float t = -log(1 - random()) / m; // Free flight
        x += w * min(t, md);
        if (t > md)
        {
            d -= md;
            continue;
        }
        if (t > d) break;
        float s = sigma(object, x);
        float Pc = s / m;

        if (random() >= Pc) // assume null collision
        {
            d -= t;
            continue;
        }

        vec3 sa = scattering_albedo(object, x);
        vec3 dL_dsa = dL_dW * W / max(vec3(EPS), sa);
        scattering_albedo_bw(object, x, dL_dsa);

        vec3 wo;
        float rho = phase_sampler(object, x, w, wo);

        W *= sa * rho;

        if (all(lessThan(W, vec3(EPS))))
        return;

        w = wo;
        boundary(object, x, w, tMin, d);
    }
}