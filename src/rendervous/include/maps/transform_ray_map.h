FORWARD {
    vec4 x = vec4(_input[0], _input[1], _input[2], 1.0);
    vec4 w = vec4(_input[3], _input[4], _input[5], 0.0);
    x = parameters.inverse_transform * x;
    w = parameters.inverse_transform * w;
    x.xyz /= x.w;
    forward(parameters.base_map, float[6](x.x, x.y, x.z, w.x, w.y, w.z), _output);
}

BACKWARD {
    vec4 x = vec4(_input[0], _input[1], _input[2], 1.0);
    vec4 w = vec4(_input[3], _input[4], _input[5], 0.0);
    x = parameters.inverse_transform * x;
    w = parameters.inverse_transform * w;
    x.xyz /= x.w;
    float _base_input_grad[6];
    backward(parameters.base_map, float[6](x.x, x.y, x.z, w.x, w.y, w.z), _output_grad, _base_input_grad);
    // TODO: Compute input grads here!
}