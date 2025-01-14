float transmittance(map_object, vec3 x, vec3 w)
{
    float value[1];
    forward(parameters.transmittance, float[6] ( x.x, x.y, x.z, w.x, w.y, w.z ), value);
    return value[0];
}

void transmittance_bw(map_object, vec3 x, vec3 w, float dL_dT)
{
    backward(parameters.transmittance, float[6] (x.x, x.y, x.z, w.x, w.y, w.z), float[1](dL_dT));
}

