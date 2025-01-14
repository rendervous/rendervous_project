void medium_path_integrator(map_object, vec3 x, vec3 w, float d, out vec3 xo, out vec3 wo, out vec3 W, out vec3 A)
{
    W = vec3(1.0);
    A = vec3(0.0);
    xo = d * w + x;
    wo = w;
    float majorant = max(0.00001, param_float(parameters.majorant));
    while(true)
    {
        float t = -log(1 - random()) / majorant;
        if (t > d - 0.00001) // exit
            break;
        x += t * w;
        d -= t;
        float s = sigma(object, x);
        if (random() < s / majorant) // interaction
        {
    #if MEDIUM_FILTER & 1  // Scattering is required
            xo = x;
            float weight, rho;
            wo = sample_phase(object, w, weight, rho);
            W = scattering_albedo(object, x);
    #else
            W = vec3(0.0);
    #endif
    #if MEDIUM_FILTER & 2  // Accumulation is required
            A = gathering(object, x, w);
    #endif
            break;
        }
    }
}
