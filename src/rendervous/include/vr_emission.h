vec3 emission(map_object, vec3 x, vec3 w)
{
    float value[3];
    forward(parameters.emission, float[6] (x.x, x.y, x.z, w.x, w.y, w.z), value);
    return vec3(value[0], value[1], value[2]);
}

void emission_bw(map_object, vec3 x, vec3 w, vec3 dL_demi)
{
    backward(parameters.emission, float[6] (x.x, x.y, x.z, w.x, w.y, w.z), float[3](dL_demi.x, dL_demi.y, dL_demi.z));
}
