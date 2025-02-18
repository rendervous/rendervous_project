FORWARD {
    _output = _input;
}

BACKWARD {
    for (int i=0; i<INPUT_DIM; i++)
        _input_grad[i] += _output_grad[i];
}
