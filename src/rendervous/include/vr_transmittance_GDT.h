float transmittance_GDT(map_object, vec3 x, vec3 w)
{
    float tMin, tMax;
    // Intersect with generic boundary first
    if (!boundary(object, x, w, tMin, tMax))
    return 1.0;

    // Intersect with grid boundary after
    float gMin, gMax;
    if (!intersect_ray_box(x, w, parameters.box_min, parameters.box_max, gMin, gMax))
    return 1.0;

    tMin = max(tMin, gMin);
    tMax = min(tMax, gMax);

    if (tMax <= tMin) // no intersection between boundaries
    return 1.0;

    x += w * tMin;
    float d = tMax - tMin;

    vec3 box_size = parameters.box_max - parameters.box_min;
    ivec3 dim = ivec3(parameters.grid.shape[2] - 1, parameters.grid.shape[1] - 1, parameters.grid.shape[0] - 1);
    vec3 cell_size = box_size / dim;
    ivec3 cell = ivec3((x - parameters.box_min) * dim / box_size);
    cell = clamp(cell, ivec3(0), dim - ivec3(1));
    vec3 alpha_inc = cell_size / max(vec3(0.000001), abs(w));
    ivec3 side = ivec3(sign(w));
    vec3 corner = (cell + side * 0.5 + vec3(0.5)) * cell_size + parameters.box_min;
    vec3 alpha = abs(corner - x) / max(vec3(0.000001), abs(w));
    float current_t = 0;
    vec3 vn = (x - parameters.box_min) * dim / box_size;
    vec3 vm = w * dim / box_size;

    float[2][2][2] sigma_values;

    float T = 1.0;

    while(current_t < d - 0.0001){
        float next_t = min(d, min(alpha.x, min(alpha.y, alpha.z)));

        load_cell(object, cell, sigma_values);
        float majorant = max(max(
            max (sigma_values[0][0][0], sigma_values[0][0][1]),
            max (sigma_values[0][1][0], sigma_values[0][1][1])),
            max(
            max (sigma_values[1][0][0], sigma_values[1][0][1]),
            max (sigma_values[1][1][0], sigma_values[1][1][1])));

        float minorant = min(min(
            min (sigma_values[0][0][0], sigma_values[0][0][1]),
            min (sigma_values[0][1][0], sigma_values[0][1][1])),
            min(
            min (sigma_values[1][0][0], sigma_values[1][0][1]),
            min (sigma_values[1][1][0], sigma_values[1][1][1])));

        float cell_t = current_t;
        // homogeneous Transmittance
        T *= exp(-minorant * (next_t - current_t));
        float r_maj = max(0.000001, majorant - minorant);
        while(true)
        {
            float dt = -log(1 - random()) / r_maj;
            if (cell_t + dt > next_t)
            break;
            cell_t += dt;
            vec3 interpolation_alpha = (vm * cell_t + vn) - vec3(cell);
            float sigma_value = interpolated_sigma(object, interpolation_alpha, sigma_values);
            float Pc = (sigma_value - minorant) / r_maj;
            if (random() < Pc)
                return 0.0;
        }

        ivec3 cell_inc = ivec3(
            alpha.x <= alpha.y && alpha.x <= alpha.z,
            alpha.x > alpha.y && alpha.y <= alpha.z,
            alpha.x > alpha.z && alpha.y > alpha.z);

        current_t = next_t;
        alpha += cell_inc * alpha_inc;
        cell += cell_inc * side;
    }

    return T;
}