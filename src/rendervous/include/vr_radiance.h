vec3 radiance(map_object, vec3 x, vec3 w)
{
    float value[3];
    forward(parameters.radiance, float[] (x.x, x.y, x.z, w.x, w.y, w.z), value);
    return vec3(value[0], value[1], value[2]);
}

void radiance_bw(map_object, vec3 x, vec3 w, vec3 dL_dR)
{
    backward(parameters.radiance, float[] (x.x, x.y, x.z, w.x, w.y, w.z), float[3](dL_dR.x, dL_dR.y, dL_dR.z));
}