FORWARD
{
    vec2 c = vec2(_input[0], _input[1]);
    vec2 ncoord = (c - parameters.bmin) * parameters.inv_bsize;
    if (any(lessThan(ncoord, vec2(0.0))) || any(greaterThanEqual(ncoord, vec2(1.0)))) {
        [[unroll]]
        for (int i=0; i<OUTPUT_DIM; i++)
            _output[i] = 0.0;
        return;
    }
    vec2 grid_coord = ncoord * vec2(parameters.grid.shape[1] - 1, parameters.grid.shape[0] - 1);
    ivec2 p = ivec2(floor(grid_coord));
    vec2 alpha = grid_coord - p;
    float_ptr grid_buf_00 = param_buffer(parameters.grid, p + ivec2(0,0));
    float_ptr grid_buf_01 = param_buffer(parameters.grid, p + ivec2(1,0));
    float_ptr grid_buf_10 = param_buffer(parameters.grid, p + ivec2(0,1));
    float_ptr grid_buf_11 = param_buffer(parameters.grid, p + ivec2(1,1));
    [[unroll]]
    for (int i=0; i<OUTPUT_DIM; i++)
        _output[i] = mix(
            mix(grid_buf_00.data[i], grid_buf_01.data[i], alpha.x),
            mix(grid_buf_10.data[i], grid_buf_11.data[i], alpha.x), alpha.y
        );
}
