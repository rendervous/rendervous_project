

void environment_sampler_pdf(map_object, vec3 x, vec3 we, out float pdf)
{
    float[1] _output;
    forward(parameters.environment_sampler_pdf, float[6](x.x, x.y, x.z, we.x, we.y, we.z), _output);
    pdf = _output[0];
}