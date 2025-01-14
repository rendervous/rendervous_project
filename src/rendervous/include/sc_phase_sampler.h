

vec3 sample_phase(map_object, vec3 win, out float weight, out float pdf)
{
    float _output[5];
    forward(parameters.phase_sampler, float[3](win.x, win.y, win.z), _output);
    weight = _output[3];
    pdf = _output[4];
    return vec3(_output[0], _output[1], _output[2]);
}