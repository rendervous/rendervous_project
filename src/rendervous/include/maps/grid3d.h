void blend(map_object, inout float dst[OUTPUT_DIM], float_ptr src, float alpha)
{
    [[unroll]] for (int i=0; i<OUTPUT_DIM; i++) dst[i] += src.data[i] * alpha;
}

FORWARD
{
    vec3 c = vec3(_input[0], _input[1], _input[2]);
    vec3 ncoord = (c - parameters.bmin) * parameters.inv_bsize;
    [[unroll]]
    for (int i=0; i<OUTPUT_DIM; i++)
        _output[i] = 0.0;
    if (any(lessThan(ncoord, vec3(0.0))) || any(greaterThanEqual(ncoord, vec3(1.0))))
    {
        return;
    }
    ivec3 dim = ivec3(parameters.grid.shape[2] - 1, parameters.grid.shape[1] - 1, parameters.grid.shape[0] - 1);
    vec3 grid_coord = ncoord * vec3(dim);
    vec3 alpha = fract(grid_coord);
    ivec3 p = clamp(ivec3(grid_coord), ivec3(0), dim - 1);
    float_ptr g = param_buffer(parameters.grid, p + ivec3(0,0,0));
    blend(object, _output, g, (1 - alpha.x)*(1 - alpha.y)*(1 - alpha.z));
    g = param_buffer(parameters.grid, p + ivec3(1,0,0));
    blend(object, _output, g, alpha.x*(1 - alpha.y)*(1 - alpha.z));
    g = param_buffer(parameters.grid, p + ivec3(0,1,0));
    blend(object, _output, g, (1 - alpha.x)*alpha.y*(1 - alpha.z));
    g = param_buffer(parameters.grid, p + ivec3(1,1,0));
    blend(object, _output, g, alpha.x*alpha.y*(1 - alpha.z));
    g = param_buffer(parameters.grid, p + ivec3(0,0,1));
    blend(object, _output, g, (1 - alpha.x)*(1 - alpha.y)*alpha.z);
    g = param_buffer(parameters.grid, p + ivec3(1,0,1));
    blend(object, _output, g, alpha.x*(1 - alpha.y)*alpha.z);
    g = param_buffer(parameters.grid, p + ivec3(0,1,1));
    blend(object, _output, g, (1 - alpha.x)*alpha.y*alpha.z);
    g = param_buffer(parameters.grid, p + ivec3(1,1,1));
    blend(object, _output, g, alpha.x*alpha.y*alpha.z);
}

BACKWARD
{
    if (parameters.grid.grad_data == 0) // NULL GRAD
        return;

    vec3 c = vec3(_input[0], _input[1], _input[2]);
    vec3 ncoord = (c - parameters.bmin) * parameters.inv_bsize;

    if (any(lessThan(ncoord, vec3(0.0))) || any(greaterThanEqual(ncoord, vec3(1.0))))
        return;

    ivec3 dim = ivec3(parameters.grid.shape[2] - 1, parameters.grid.shape[1] - 1, parameters.grid.shape[0] - 1);
    vec3 grid_coord = ncoord * vec3(dim);
    vec3 alpha = fract(grid_coord);
    ivec3 p = clamp(ivec3(grid_coord), ivec3(0), dim - 1);
    float_ptr g000 = param_grad_buffer(parameters.grid, p + ivec3(0,0,0));
    float_ptr g001 = param_grad_buffer(parameters.grid, p + ivec3(1,0,0));
    float_ptr g010 = param_grad_buffer(parameters.grid, p + ivec3(0,1,0));
    float_ptr g011 = param_grad_buffer(parameters.grid, p + ivec3(1,1,0));
    float_ptr g100 = param_grad_buffer(parameters.grid, p + ivec3(0,0,1));
    float_ptr g101 = param_grad_buffer(parameters.grid, p + ivec3(1,0,1));
    float_ptr g110 = param_grad_buffer(parameters.grid, p + ivec3(0,1,1));
    float_ptr g111 = param_grad_buffer(parameters.grid, p + ivec3(1,1,1));

    [[unroll]]
    for (int i=0; i<OUTPUT_DIM; i++)
    {
        atomicAdd_f(g000, i, _output_grad[i] * (1 - alpha.x) * (1 - alpha.y) * (1 - alpha.z));
        atomicAdd_f(g001, i, _output_grad[i] * (alpha.x) * (1 - alpha.y) * (1 - alpha.z));
        atomicAdd_f(g010, i, _output_grad[i] * (1 - alpha.x) * (alpha.y) * (1 - alpha.z));
        atomicAdd_f(g011, i, _output_grad[i] * (alpha.x) * (alpha.y) * (1 - alpha.z));
        atomicAdd_f(g100, i, _output_grad[i] * (1 - alpha.x) * (1 - alpha.y) * (alpha.z));
        atomicAdd_f(g101, i, _output_grad[i] * (alpha.x) * (1 - alpha.y) * (alpha.z));
        atomicAdd_f(g110, i, _output_grad[i] * (1 - alpha.x) * (alpha.y) * (alpha.z));
        atomicAdd_f(g111, i, _output_grad[i] * (alpha.x) * (alpha.y) * (alpha.z));
    }
}