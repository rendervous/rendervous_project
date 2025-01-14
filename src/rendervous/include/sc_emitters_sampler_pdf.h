

void emitters_sampler_pdf(map_object, vec3 x, vec3 fN, vec3 we, out float pdf)
{
    float _input[9] = float[9](x.x, x.y, x.z, fN.x, fN.y, fN.z, we.x, we.y, we.z);
    float _output[1];
    forward(parameters.emitters_sampler_pdf, _input, _output);
    pdf = _output[0];
}