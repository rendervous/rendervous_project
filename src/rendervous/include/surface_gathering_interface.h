void surface_gathering(map_object, vec3 win, Surfel surfel, out vec3 A);

void surface_gathering_bw(map_object, vec3 win, Surfel surfel, vec3 out_A, vec3 dL_dA);

FORWARD {
    vec3 win = vec3(_input[0], _input[1], _input[2]);
    Surfel surfel = Surfel(
        vec3(_input[3], _input[4], _input[5]),
        vec3(_input[6], _input[7], _input[8]),
        vec3(_input[9], _input[10], _input[11]),
        vec2(_input[12], _input[13]),
        vec3(_input[14], _input[15], _input[16]),
        vec3(_input[17], _input[18], _input[19]));
    vec3 A;
    surface_gathering(object, win, surfel, A);
    _output = float[3](A.x, A.y, A.z);
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
    vec3 dL_dA  = vec3(_output_grad[0], _output_grad[1], _output_grad[2]);
    vec3 out_A = vec3(_output[0], _output[1], _output[2]);
    surface_gathering_bw(object, win, surfel, out_A, dL_dA);
    // TODO: Include _input_grads!
}