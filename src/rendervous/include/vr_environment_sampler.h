vec3 environment_sampler(map_object, vec3 x, vec3 w, out vec3 wo)
{
    float value[6];
    forward(parameters.environment_sampler, float[](x.x, x.y, x.z, w.x, w.y, w.z), value);
    wo = vec3(value[3], value[4], value[5]);
    return vec3(value[0], value[1], value[2]);
}

void environment_sampler_bw(map_object, vec3 x, vec3 w, vec3 dL_denv)
{
    backward(parameters.environment_sampler, float[](x.x, x.y, x.z, w.x, w.y, w.z), float[](dL_denv.x, dL_denv.y, dL_denv.z, 0.0, 0.0, 0.0));
}
