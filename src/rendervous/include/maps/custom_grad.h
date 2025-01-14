FORWARD {
    forward(parameters.fw, _input, _output);
}

BACKWARD {
    backward(parameters.bw, _input, _output_grad, _input_grad);
}

BACKWARD_USING_OUTPUT {
    backward(parameters.bw, _input, _output, _output_grad, _input_grad);
}