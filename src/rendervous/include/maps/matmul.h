REQUIRES_MATMUL(INPUT_DIM, OUTPUT_DIM)

FORWARD {
    if (parameters.is_pre != 0)
        pre_matmul_fw(object, _input, _output, parameters.matrix.data);
    else
        matmul_fw(object, _input, _output, parameters.matrix.data);
}

BACKWARD {
    if (parameters.is_pre != 0)
        pre_matmul_bw(object, _input, _output_grad, _input_grad, parameters.matrix.data, parameters.matrix.grad_data);
    else
        matmul_bw(object, _input, _output_grad, _input_grad, parameters.matrix.data, parameters.matrix.grad_data);
}