FORWARD
{
    vec2 c = vec2(_input[0], _input[1]);
    vec2 ncoord = (c - parameters.bmin) * parameters.inv_bsize;
    ncoord = clamp(ncoord, vec2(0.0), vec2(1.0));
//    if (any(lessThan(ncoord, vec2(0.0))) || any(greaterThanEqual(ncoord, vec2(1.0)))) {
//        [[unroll]]
//        for (int i=0; i<OUTPUT_DIM; i++)
//            _output[i] = 0.0;
//        return;
//    }
    vec2 grid_coord = (ncoord * vec2(parameters.grid.shape[1], parameters.grid.shape[0]) - vec2(0.5));
    ivec2 max_dim = ivec2(parameters.grid.shape[1], parameters.grid.shape[0]) - 1;
    ivec2 p = ivec2(floor(grid_coord));
    vec2 alpha = grid_coord - p;
    float_ptr grid_buf_00 = param_buffer(parameters.grid, clamp(p + ivec2(0,0), ivec2(0), max_dim));
    float_ptr grid_buf_01 = param_buffer(parameters.grid, clamp(p + ivec2(1,0), ivec2(0), max_dim));
    float_ptr grid_buf_10 = param_buffer(parameters.grid, clamp(p + ivec2(0,1), ivec2(0), max_dim));
    float_ptr grid_buf_11 = param_buffer(parameters.grid, clamp(p + ivec2(1,1), ivec2(0), max_dim));
    [[unroll]]
    for (int i=0; i<OUTPUT_DIM; i++)
        _output[i] = mix(
            mix(grid_buf_00.data[i], grid_buf_01.data[i], alpha.x),
            mix(grid_buf_10.data[i], grid_buf_11.data[i], alpha.x), alpha.y
        );
}

BACKWARD
{
    if (parameters.grid.grad_data == 0) // NULL GRAD
        return;

    vec2 c = vec2(_input[0], _input[1]);
    vec2 ncoord = (c - parameters.bmin) * parameters.inv_bsize;
    ncoord = clamp(ncoord, vec2(0.0), vec2(1.0));

    vec2 grid_coord = (ncoord * vec2(parameters.grid.shape[1], parameters.grid.shape[0]) - vec2(0.5));
    ivec2 max_dim = ivec2(parameters.grid.shape[1], parameters.grid.shape[0]) - 1;
    ivec2 p = ivec2(floor(grid_coord));
    vec2 alpha = grid_coord - p;
    float_ptr g00 = param_grad_buffer(parameters.grid, clamp(p + ivec2(0,0), ivec2(0), max_dim));
    float_ptr g01 = param_grad_buffer(parameters.grid, clamp(p + ivec2(1,0), ivec2(0), max_dim));
    float_ptr g10 = param_grad_buffer(parameters.grid, clamp(p + ivec2(0,1), ivec2(0), max_dim));
    float_ptr g11 = param_grad_buffer(parameters.grid, clamp(p + ivec2(1,1), ivec2(0), max_dim));

    [[unroll]]
    for (int i=0; i<OUTPUT_DIM; i++)
    {
        atomicAdd_f(g00, i, _output_grad[i] * (1 - alpha.x) * (1 - alpha.y));
        atomicAdd_f(g01, i, _output_grad[i] * (alpha.x) * (1 - alpha.y));
        atomicAdd_f(g10, i, _output_grad[i] * (1 - alpha.x) * (alpha.y));
        atomicAdd_f(g11, i, _output_grad[i] * (alpha.x) * (alpha.y));
    }
}