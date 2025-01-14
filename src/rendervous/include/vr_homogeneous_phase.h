float homogeneous_phase(map_object, vec3 w, vec3 wo)
{
    float value[1];
    forward(parameters.homogeneous_phase, float[] (w.x, w.y, w.z, wo.x, wo.y, wo.z), value);
    return value[0];
}

void homogeneous_phase_bw(map_object, vec3 w, vec3 wo, float dL_dph)
{
    backward(parameters.homogeneous_phase, float[] (w.x, w.y, w.z, wo.x, wo.y, wo.z), float[](dL_dph));
}
