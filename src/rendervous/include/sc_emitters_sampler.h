
void emitters_sampler(map_object, vec3 x, vec3 fN, out vec3 we, out vec3 E, out float pdf)
{
    float _input[6] = float[6](x.x, x.y, x.z, fN.x, fN.y, fN.z);
    float _output[7];
    forward(parameters.emitters_sampler_pdf, _input, _output);
    we = vec3(_output[0], _output[1], _output[2]);
    E = vec3(_output[3], _output[4], _output[5]);
    pdf = _output[6];
}