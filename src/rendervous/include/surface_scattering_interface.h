void surface_scattering(map_object, vec3 win, vec3 wout, Surfel surfel, out vec3 W);

void surface_scattering_bw(map_object, vec3 win, vec3 wout, Surfel surfel, vec3 dL_dW);

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
    vec3 W;
    surface_scattering(object, win, wout, surfel, W);
    _output = float[3](W.x, W.y, W.z);
}

BACKWARD {
    vec3 win = vec3(_input[0], _input[1], _input[2]);
    vec3 wout = vec3(_input[3], _input[4], _input[5]);
    Surfel surfel = Surfel(
        vec3(_input[6], _input[7], _input[8]),
        vec3(_input[9], _input[10], _input[11]),
        vec3(_input[12], _input[13], _input[14]),
        vec2(_input[15], _input[16]),
        vec3(_input[17], _input[18], _input[19]),
        vec3(_input[20], _input[21], _input[22]));
    vec3 dL_dW = vec3(_output_grad[0], _output_grad[1], _output_grad[2]);
    surface_scattering_bw(object, win, wout, surfel, dL_dW);
    // TODO: Include _input_grads!
}