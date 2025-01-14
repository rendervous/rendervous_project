void surface_scattering_sampler_pdf(map_object, vec3 win, vec3 wout, Surfel surfel, out float pdf);

FORWARD {
    vec3 win = vec3(_input[0], _input[1], _input[2]);
    vec3 wout = vec3(_input[3], _input[4], _input[5]);
    Surfel surfel = Surfel(
        vec3(_input[6], _input[7], _input[8]),
        vec3(_input[9], _input[10], _input[11]),
        vec3(_input[12], _input[13], _input[14]),
        vec2(_input[15], _input[16]),
        vec3(_input[17], _input[18], _input[19]),
        vec3(_input[20], _input[21], _input[22]));
    float pdf;
    surface_scattering_sampler_pdf(object, win, wout, surfel, pdf);
    _output = float[1](pdf);
}
