

void environment_sampler(map_object, vec3 x, out vec3 we, out vec3 E, out float pdf)
{
    float[7] _output;
    forward(parameters.environment_sampler, float[3](x.x, x.y, x.z), _output);
    we = vec3(_output[0], _output[1], _output[2]);
    E = vec3(_output[3], _output[4], _output[5]);
    pdf = _output[6];
}