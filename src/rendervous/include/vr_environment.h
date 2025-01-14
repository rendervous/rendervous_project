vec3 environment(map_object, vec3 w)
{
    float value[3];
    forward(parameters.environment, float[3] (w.x, w.y, w.z), value);
    return vec3(value[0], value[1], value[2]);
}

void environment_bw(map_object, vec3 w, vec3 dL_denv)
{
    backward(parameters.environment, float[3] (w.x, w.y, w.z), float[3](dL_denv.x, dL_denv.y, dL_denv.z));
}