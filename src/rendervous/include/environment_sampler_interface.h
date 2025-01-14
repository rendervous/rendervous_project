void environment_sampler(map_object, vec3 x, out vec3 w, out vec3 E, out float pdf);

void environment_sampler_bw(map_object, vec3 x, vec3 out_w, vec3 out_E, vec3 dL_dw, vec3 dL_dE);

FORWARD {
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w;
    vec3 E;
    float pdf;
    environment_sampler(object, x, w, E, pdf);
    _output = float[7](w.x, w.y, w.z, E.x, E.y, E.z, pdf);
}

BACKWARD_USING_OUTPUT {
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 out_w = vec3(_output[0], _output[1], _output[2]);
    vec3 out_E = vec3(_output[3], _output[4], _output[5]);
    vec3 dL_dw = vec3(_output_grad[0], _output_grad[1], _output_grad[2]);
    vec3 dL_dE = vec3(_output_grad[3], _output_grad[4], _output_grad[5]);
    environment_sampler_bw(object, x, out_w, out_E, dL_dw, dL_dE);
}