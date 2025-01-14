FORWARD {
    float output_map[MAP_OUTPUT_DIM];
    forward(parameters.map, _input, output_map);
    [[unroll]] for (int i=0; i < MAP_OUTPUT_DIM; i++) _output[i] = output_map[parameters.indices[i]];
}

BACKWARD {
    float output_map_grad[MAP_OUTPUT_DIM];
    [[unroll]] for (int i=0; i < MAP_OUTPUT_DIM; i++) output_map_grad[i] = 0.0;
    [[unroll]] for (int i=0; i < OUTPUT_DIM; i++) output_map_grad[parameters.indices[i]] += _output_grad[i];
    backward(parameters.map, _input, output_map_grad, _input_grad);
}