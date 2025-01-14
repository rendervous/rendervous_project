FORWARD {
    float intermediate [INTERMEDIATE_DIM];
    forward(parameters.inner, _input, intermediate);
    forward(parameters.outter, intermediate, _output);
}

BACKWARD {
    float intermediate [INTERMEDIATE_DIM];
    forward(parameters.inner, _input, intermediate);
    float intermediate_grad [INTERMEDIATE_DIM];
    [[unroll]] for (int i=0; i<INTERMEDIATE_DIM; i++) intermediate_grad[i] = 0.0;
    backward(parameters.outter, intermediate, _output_grad, intermediate_grad);
    // backward(parameters.inner, _input, intermediate, intermediate_grad, _input_grad);
    backward(parameters.inner, _input, intermediate_grad, _input_grad);
}