FORWARD {
    float _temp[OUTPUT_DIM];
    forward(parameters.map_a, _input, _output);
    forward(parameters.map_b, _input, _temp);
    [[unroll]]
    for (int i=0; i<OUTPUT_DIM; i++)
        _output[i] *= _temp[i];
}

BACKWARD {
    float a[OUTPUT_DIM];
    forward(parameters.map_a, _input, a);
    float b[OUTPUT_DIM];
    forward(parameters.map_b, _input, b);
    float dL_darg[OUTPUT_DIM];
    [[unroll]] for (int i=0; i<OUTPUT_DIM; i++) dL_darg[i] = _output_grad[i] * b[i];
    backward(parameters.map_a, _input, a, dL_darg, _input_grad); // output provided just in case
    [[unroll]] for (int i=0; i<OUTPUT_DIM; i++) dL_darg[i] = _output_grad[i] * a[i];
    backward(parameters.map_b, _input, b, dL_darg, _input_grad); // output provided just in case
}