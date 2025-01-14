float phase(map_object, vec3 x, vec3 w, vec3 wo)
{
    float value[1];
    forward(parameters.phase, float[] (x.x, x.y, x.z, w.x, w.y, w.z, wo.x, wo.y, wo.z), value);
    return value[0];
}

void phase_bw(map_object, vec3 x, vec3 w, vec3 wo, float dL_dph)
{
    backward(parameters.phase, float[] (x.x, x.y, x.z, w.x, w.y, w.z, wo.x, wo.y, wo.z), float[](dL_dph));
}
