float transmittance_RM(map_object, vec3 x, vec3 w)
{
    float tMin, tMax;
    if (!boundary(object, x, w, tMin, tMax))
    return 1.0;

    x += w * tMin;
    float d = tMax - tMin;

    int samples = int((d - 0.000001) / parameters.step + 1);

    float tau = 0;
    for (int i=0; i<samples; i++)
    {
        float t = (i + random()) * parameters.step;
        vec3 xt = w * t + x;
        tau += sigma(object, xt) * parameters.step;
    }

    return exp(-tau);
}