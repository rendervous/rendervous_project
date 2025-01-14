float sigma_at(map_object, ivec3 cell)
{
    ivec3 dim = ivec3(parameters.grid.shape[2] - 1, parameters.grid.shape[1] - 1, parameters.grid.shape[0] - 1);
    cell = clamp(cell, ivec3(0), dim);
    float_ptr buf = param_buffer(parameters.grid, cell);
    return buf.data[0];
}

void load_cell(map_object, ivec3 cell, out float[2][2][2] sigmas)
{
    [[unroll]]
    for (int dz = 0; dz < 2; dz ++)
        [[unroll]]
        for (int dy = 0; dy < 2; dy ++)
            [[unroll]]
            for (int dx = 0; dx < 2; dx ++)
                sigmas[dz][dy][dx] = sigma_at(object, cell + ivec3(dx, dy, dz));
}

float interpolated_sigma(map_object, vec3 alpha, float[2][2][2] sigmas)
{
    return mix(mix(
        mix(sigmas[0][0][0], sigmas[0][0][1], alpha.x),
        mix(sigmas[0][1][0], sigmas[0][1][1], alpha.x), alpha.y),
        mix(
        mix(sigmas[1][0][0], sigmas[1][0][1], alpha.x),
        mix(sigmas[1][1][0], sigmas[1][1][1], alpha.x), alpha.y), alpha.z);
}

void interpolated_sigma_bw(map_object, vec3 alpha, float dL_dsigma, inout float[2][2][2] dL_dcell)
{
    dL_dcell[0][0][0] = dL_dsigma * (1 - alpha.x)*(1 - alpha.y)*(1 - alpha.z);
    dL_dcell[0][0][1] = dL_dsigma * (alpha.x)*(1 - alpha.y)*(1 - alpha.z);
    dL_dcell[0][1][0] = dL_dsigma * (1 - alpha.x)*(alpha.y)*(1 - alpha.z);
    dL_dcell[0][1][1] = dL_dsigma * (alpha.x)*(alpha.y)*(1 - alpha.z);
    dL_dcell[1][0][0] = dL_dsigma * (1 - alpha.x)*(1 - alpha.y)*(alpha.z);
    dL_dcell[1][0][1] = dL_dsigma * (alpha.x)*(1 - alpha.y)*(alpha.z);
    dL_dcell[1][1][0] = dL_dsigma * (1 - alpha.x)*(alpha.y)*(alpha.z);
    dL_dcell[1][1][1] = dL_dsigma * (alpha.x)*(alpha.y)*(alpha.z);
}

void add_interpolated_sigma_bw(map_object, vec3 alpha, float dL_dsigma, inout float[2][2][2] dL_dcell)
{
    dL_dcell[0][0][0] += dL_dsigma * (1 - alpha.x)*(1 - alpha.y)*(1 - alpha.z);
    dL_dcell[0][0][1] += dL_dsigma * (alpha.x)*(1 - alpha.y)*(1 - alpha.z);
    dL_dcell[0][1][0] += dL_dsigma * (1 - alpha.x)*(alpha.y)*(1 - alpha.z);
    dL_dcell[0][1][1] += dL_dsigma * (alpha.x)*(alpha.y)*(1 - alpha.z);
    dL_dcell[1][0][0] += dL_dsigma * (1 - alpha.x)*(1 - alpha.y)*(alpha.z);
    dL_dcell[1][0][1] += dL_dsigma * (alpha.x)*(1 - alpha.y)*(alpha.z);
    dL_dcell[1][1][0] += dL_dsigma * (1 - alpha.x)*(alpha.y)*(alpha.z);
    dL_dcell[1][1][1] += dL_dsigma * (alpha.x)*(alpha.y)*(alpha.z);
}

bool sigma_grid_has_grad(map_object)
{
    return parameters.grid.grad_data != 0;
}

void sigma_at_bw(map_object, ivec3 cell, float dL_dg)
{
    ivec3 dim = ivec3(parameters.grid.shape[2] - 1, parameters.grid.shape[1] - 1, parameters.grid.shape[0] - 1);
    cell = clamp(cell, ivec3(0), dim);
    float_ptr buf = param_grad_buffer(parameters.grid, cell);
    atomicAdd_f(buf, 0, dL_dg);
}

void update_cell_gradients(map_object, ivec3 cell, float[2][2][2] dL_dcell)
{
    [[unroll]]
    for (int dz = 0; dz < 2; dz ++)
        [[unroll]]
        for (int dy = 0; dy < 2; dy ++)
            [[unroll]]
            for (int dx = 0; dx < 2; dx ++)
                sigma_at_bw(object, cell + ivec3(dx, dy, dz), dL_dcell[dz][dy][dx]);
}