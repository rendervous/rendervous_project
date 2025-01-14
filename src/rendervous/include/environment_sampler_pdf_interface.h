
void environment_sampler_pdf(map_object, vec3 x, vec3 w, out float pdf);

FORWARD {
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    float pdf;
    environment_sampler_pdf(object, x, w, pdf);
    _output = float[1](pdf);
}
