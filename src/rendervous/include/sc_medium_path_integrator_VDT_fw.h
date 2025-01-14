void medium_path_integrator(map_object, vec3 x, vec3 w, float d, out vec3 xo, out vec3 wo, out vec3 W, out vec3 A)
{
    W = vec3(1.0);
    A = vec3(0.0);
    float majorant = max(0.00001, param_float(parameters.majorant));
    while(true)
    {
        float t = -log(1 - random()) / majorant;
        if (t > d - 0.00001) // exit
            break;
        x += t * w;
        d -= t;
        float s = sigma(object, x);// + 1e-10;
        float Pc = s / majorant;
        if (random() < Pc) // interaction
        {
    #if MEDIUM_FILTER & 2  // Accumulation is required
            A += W * gathering(object, x, w);
    #endif
    #if MEDIUM_FILTER & 1  // Scattering is required
            float weight, pdf_wo;
            w = sample_phase(object, w, weight, pdf_wo);
            W *= scattering_albedo(object, x) * weight;
            // recompute distance towards new direction w
            if (!raycast(object, x, w, d))
            break;
    #else
            W = vec3(0.0);
            break;
    #endif
        }
    }
    wo = w;
    xo = w*d + x;
}