FORWARD {
    float _temp[OUTPUT_DIM];
    forward(parameters.map_a, _input, _output);
    forward(parameters.map_b, _input, _temp);
    [[unroll]]
    for (int i=0; i<OUTPUT_DIM; i++)
        _output[i] -= _temp[i];
}

BACKWARD {
    backward(parameters.map_a, _input, _output_grad, _input_grad);
    [[unroll]] for (int i=0; i<OUTPUT_DIM; i++) _output_grad[i] *= -1;
    backward(parameters.map_b, _input, _output_grad, _input_grad);
}
