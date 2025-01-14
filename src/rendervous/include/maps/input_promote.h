FORWARD {
    float output_map[OUTPUT_DIM];
    [[unroll]] for (int i=0; i < OUTPUT_DIM; i++) _output[i] = _input[0];
}

BACKWARD {
    [[unroll]] for (int i=0; i < OUTPUT_DIM; i++) _input_grad[0] += _output_grad[i];
}

