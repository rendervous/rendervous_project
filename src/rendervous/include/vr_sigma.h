float sigma(map_object, vec3 x)
{
    float value[1];
    forward(parameters.sigma, float[3] (x.x, x.y, x.z), value);
    return value[0];
}

void sigma_bw(map_object, vec3 x, float dL_dsigma)
{
    backward(parameters.sigma, float[3] (x.x, x.y, x.z), float[1](dL_dsigma));
}
