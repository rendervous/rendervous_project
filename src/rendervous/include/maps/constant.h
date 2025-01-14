FORWARD {
    float_ptr data_ptr = float_ptr(parameters.value.data);
    [[unroll]] for (int i=0; i < OUTPUT_DIM; i++) _output[i] = data_ptr.data[i];
}

BACKWARD {
    if (parameters.value.grad_data == 0) return;
    float_ptr grad_data = float_ptr(parameters.value.grad_data);
    [[unroll]] for (int i=0; i<OUTPUT_DIM; i++)
        atomicAdd_f(grad_data, i, _output_grad[i]);
}