void surface_scattering_sampler(map_object, vec3 win, Surfel surfel, out vec3 wout, out vec3 W, out float pdf);

void surface_scattering_sampler_bw(map_object, vec3 win, Surfel surfel, vec3 out_W, vec3 dL_dwout, vec3 dL_dW);

FORWARD {
    vec3 win = vec3(_input[0], _input[1], _input[2]);
    Surfel surfel = Surfel(
        vec3(_input[3], _input[4], _input[5]),
        vec3(_input[6], _input[7], _input[8]),
        vec3(_input[9], _input[10], _input[11]),
        vec2(_input[12], _input[13]),
        vec3(_input[14], _input[15], _input[16]),
        vec3(_input[17], _input[18], _input[19]));
    vec3 wout, W; float pdf;
    surface_scattering_sampler(object, win, surfel, wout, W, pdf);
    _output = float[7](wout.x, wout.y, wout.z, W.x, W.y, W.z, pdf);
}

BACKWARD_USING_OUTPUT {
    vec3 win = vec3(_input[0], _input[1], _input[2]);
    Surfel surfel = Surfel(
        vec3(_input[3], _input[4], _input[5]),
        vec3(_input[6], _input[7], _input[8]),
        vec3(_input[9], _input[10], _input[11]),
        vec2(_input[12], _input[13]),
        vec3(_input[14], _input[15], _input[16]),
        vec3(_input[17], _input[18], _input[19]));
    vec3 out_W = vec3(_output[3], _output[4], _output[5]);
    vec3 dL_dwout = vec3(_output_grad[0], _output_grad[1], _output_grad[2]);
    vec3 dL_dW = vec3(_output_grad[3], _output_grad[4], _output_grad[5]);
    surface_scattering_sampler_bw(object, win, surfel, out_W, dL_dwout, dL_dW);
    // TODO: Include _input_grads!
}