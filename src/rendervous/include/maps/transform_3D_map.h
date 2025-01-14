FORWARD {
    vec4 x = vec4(_input[0], _input[1], _input[2], 1.0);
    x = parameters.inverse_transform * x;
    x.xyz /= x.w;
    forward(parameters.base_map, float[3](x.x, x.y, x.z), _output);
}

BACKWARD {
    vec4 x = vec4(_input[0], _input[1], _input[2], 1.0);
    x = parameters.inverse_transform * x;
    x.xyz /= x.w;
    float _base_input_grad[3];
    backward(parameters.base_map, float[3](x.x, x.y, x.z), _output_grad, _base_input_grad);
    // TODO: Compute input grads here!
}