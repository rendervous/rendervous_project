float phase_sampler(map_object, vec3 x, vec3 w, out vec3 wo)
{
    float value[4];
    forward(parameters.phase_sampler, float[](x.x, x.y, x.z, w.x, w.y, w.z), value);
    wo = vec3(value[1], value[2], value[3]);
    return value[0];
}

void phase_sampler_bw(map_object, vec3 x, vec3 w, float dL_dph)
{
    backward(parameters.phase_sampler, float[](x.x, x.y, x.z, w.x, w.y, w.z), float[](dL_dph, 0.0, 0.0, 0.0));
}