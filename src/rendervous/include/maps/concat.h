FORWARD {
    float output_a[A_OUTPUT_DIM];
    forward(parameters.map_a, _input, output_a);
    float output_b[B_OUTPUT_DIM];
    forward(parameters.map_b, _input, output_b);
    [[unroll]] for (int i=0; i < A_OUTPUT_DIM; i++) _output[i] = output_a[i];
    [[unroll]] for (int i=0; i < B_OUTPUT_DIM; i++) _output[i + A_OUTPUT_DIM] = output_b[i];
}

BACKWARD {
    float output_a_grad[A_OUTPUT_DIM];
    [[unroll]] for (int i=0; i < A_OUTPUT_DIM; i++) output_a_grad[i] = _output_grad[i];
    backward(parameters.map_a, _input, output_a_grad, _input_grad);
    float output_b_grad[B_OUTPUT_DIM];
    [[unroll]] for (int i=0; i < B_OUTPUT_DIM; i++) output_b_grad[i] = _output_grad[i + A_OUTPUT_DIM];
    backward(parameters.map_b, _input, output_b_grad, _input_grad);
}

BACKWARD_USING_OUTPUT {
    float output_a_grad[A_OUTPUT_DIM];
    float output_a[A_OUTPUT_DIM];
    [[unroll]] for (int i=0; i < A_OUTPUT_DIM; i++) { output_a_grad[i] = _output_grad[i]; output_a[i] = _output[i]; }
    backward(parameters.map_a, _input, output_a, output_a_grad, _input_grad);
    float output_b_grad[B_OUTPUT_DIM];
    float output_b[B_OUTPUT_DIM];
    [[unroll]] for (int i=0; i < B_OUTPUT_DIM; i++) { output_b_grad[i] = _output_grad[i + A_OUTPUT_DIM]; output_b[i] = _output[i + A_OUTPUT_DIM]; }
    backward(parameters.map_b, _input, output_b, output_b_grad, _input_grad);
}