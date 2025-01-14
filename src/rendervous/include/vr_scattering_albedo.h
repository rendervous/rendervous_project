vec3 scattering_albedo(map_object, vec3 x)
{
    float value[3];
    forward(parameters.scattering_albedo, float[3] (x.x, x.y, x.z), value);
    return vec3(value[0], value[1], value[2]);
}

void scattering_albedo_bw(map_object, vec3 x, vec3 dL_dsa)
{
    backward(parameters.scattering_albedo, float[3] (x.x, x.y, x.z), float[3](dL_dsa.x, dL_dsa.y, dL_dsa.z));
}