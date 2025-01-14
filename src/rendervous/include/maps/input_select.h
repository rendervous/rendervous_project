FORWARD {
    float output_map[OUTPUT_DIM];
    [[unroll]] for (int i=0; i < OUTPUT_DIM; i++) _output[i] = _input[parameters.indices[i]];
}
BACKWARD {
    [[unroll]] for (int i=0; i < OUTPUT_DIM; i++) _input_grad[parameters.indices[i]] += _output_grad[i];
}