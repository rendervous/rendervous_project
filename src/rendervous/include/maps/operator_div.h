FORWARD {
    float _temp[OUTPUT_DIM];
    forward(parameters.map_a, _input, _output);
    forward(parameters.map_b, _input, _temp);
    [[unroll]]
    for (int i=0; i<OUTPUT_DIM; i++)
        _output[i] /= _temp[i];
}

BACKWARD {
    float a[OUTPUT_DIM];
    forward(parameters.map_a, _input, a);
    float b[OUTPUT_DIM];
    forward(parameters.map_b, _input, b);
    [[unroll]] for (int i=0; i<OUTPUT_DIM; i++) _output_grad[i] /= b[i];
    backward(parameters.map_a, _input, a, _output_grad, _input_grad);
    [[unroll]] for (int i=0; i<OUTPUT_DIM; i++) _output_grad[i] *= a[i] / b[i];
    backward(parameters.map_b, _input, b, _output_grad, _input_grad);
}