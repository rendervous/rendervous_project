FORWARD {
    float r[1];
    forward(parameters.map, _input, r);
    [[unroll]]
    for (int i=0; i<OUTPUT_DIM; i++)
        _output[i] = r[0];
}

BACKWARD {
    float dL_dr[1];
    dL_dr[0] = 0.0;
    [[unroll]]
    for (int i=0; i<OUTPUT_DIM; i++)
        dL_dr[0] += _output_grad[i];
    backward(parameters.map, _input, dL_dr, _input_grad);
}

BACKWARD_USING_OUTPUT { // Internal map could make advantage of precomputed output
    float dL_dr[1];
    dL_dr[0] = 0.0;
    [[unroll]]
    for (int i=0; i<OUTPUT_DIM; i++)
        dL_dr[0] += _output_grad[i];
    backward(parameters.map, _input, float[1](_output[0]), dL_dr, _input_grad);
}