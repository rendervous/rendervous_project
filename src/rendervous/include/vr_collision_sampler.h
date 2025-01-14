float collision_sampler(map_object, vec3 x, vec3 w, out float t, out float T)
{
    float value[3];
    forward (parameters.collision_sampler, float[](x.x, x.y, x.z, w.x, w.y, w.z), value);
    t = value[1];
    T = value[2];
    return value[0];
}

void collision_sampler_bw(map_object, vec3 x, vec3 w, float dL_dC, float dL_dT)
{
    backward (parameters.collision_sampler, float[](x.x, x.y, x.z, w.x, w.y, w.z), float[] (dL_dC, 0, dL_dT));
}