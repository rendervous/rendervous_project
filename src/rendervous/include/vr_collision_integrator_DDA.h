vec3 collision_integrator_DDA(map_object, vec3 x, vec3 w)
{
    vec3 env = environment(object, w);

    float tMin, tMax;
    // Intersect with generic boundary first
    if (!boundary(object, x, w, tMin, tMax))
    return env;

    // Intersect with grid boundary after
    float gMin, gMax;
    if (!intersect_ray_box(x, w, parameters.box_min, parameters.box_max, gMin, gMax))
    return env;

    tMin = max(tMin, gMin);
    tMax = min(tMax, gMax);

    if (tMax <= tMin) // no intersection between boundaries
    return env;

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
    vec3 R = vec3(0.0); // accumulated radiance

    while(current_t < d - 0.0001){
        float next_t = min(d, min(alpha.x, min(alpha.y, alpha.z)));
        load_cell(object, cell, sigma_values);
        // TODO: Change here for a proper trilinear integration!
        float cell_t = mix(current_t, next_t, random());
        vec3 interpolation_alpha = (vm * cell_t + vn) - vec3(cell);
        float sigma_value = interpolated_sigma(object, interpolation_alpha, sigma_values);
        // homogeneous Transmittance
        float dt = next_t - current_t;
        float voxel_transmittance = exp(-dt * sigma_value);
        vec3 Le = exitance_radiance(object, x + cell_t * w, -w);
        R += T * (1 - voxel_transmittance) * Le;
        T *= voxel_transmittance;
//        if (T < 0.0001)
//        return R;

        ivec3 cell_inc = ivec3(
            alpha.x <= alpha.y && alpha.x <= alpha.z,
            alpha.x > alpha.y && alpha.y <= alpha.z,
            alpha.x > alpha.z && alpha.y > alpha.z);

        current_t = next_t;
        alpha += cell_inc * alpha_inc;
        cell += cell_inc * side;
    }

    R += T * env;

    return R;
}


void collision_integrator_DDA_bw(map_object, vec3 x, vec3 w, vec3 dL_dR)
{
    if (dL_dR == vec3(0.0))
    return;

    float tMin, tMax;
    // Intersect with generic boundary first
    if (!boundary(object, x, w, tMin, tMax))
    return;

    // Intersect with grid boundary after
    float gMin, gMax;
    if (!intersect_ray_box(x, w, parameters.box_min, parameters.box_max, gMin, gMax))
    return;

    tMin = max(tMin, gMin);
    tMax = min(tMax, gMax);

    if (tMax <= tMin) // no intersection between boundaries
    return;

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
    vec3 R = vec3(0.0); // accumulated radiance

    SAVE_SEED(before_traversal);
    while(current_t < d - 0.0001){
        float next_t = min(d, min(alpha.x, min(alpha.y, alpha.z)));
        load_cell(object, cell, sigma_values);
        // TODO: Change here for a proper trilinear integration!
        float cell_t = mix(current_t, next_t, random());
        vec3 interpolation_alpha = (vm * cell_t + vn) - vec3(cell);
        float sigma_value = interpolated_sigma(object, interpolation_alpha, sigma_values);
        // homogeneous Transmittance
        float dt = next_t - current_t;
        float voxel_transmittance = exp(-dt * sigma_value);
        vec3 Le = exitance_radiance(object, x + cell_t * w, -w);
        R += T * (1 - voxel_transmittance) * Le;
        T *= voxel_transmittance;
//        if (T < 0.0001)
//        {
//            T = 0.0;
//            break;
//        }

        ivec3 cell_inc = ivec3(
            alpha.x <= alpha.y && alpha.x <= alpha.z,
            alpha.x > alpha.y && alpha.y <= alpha.z,
            alpha.x > alpha.z && alpha.y > alpha.z);

        current_t = next_t;
        alpha += cell_inc * alpha_inc;
        cell += cell_inc * side;
    }

    vec3 env = environment(object, w);
    //R += T * environment(object, w);

    // Replay

    cell = ivec3((x - parameters.box_min) * dim / box_size);
    cell = clamp(cell, ivec3(0), dim - ivec3(1));
    alpha = abs(corner - x) / max(vec3(0.000001), abs(w));
    current_t = 0;
    float fw_T = T;
    T = 1.0;

    float[2][2][2] sigma_grad_values;

    SET_SEED(before_traversal);
    while(current_t < d - 0.0001){
        float next_t = min(d, min(alpha.x, min(alpha.y, alpha.z)));
        load_cell(object, cell, sigma_values);
        // TODO: Change here for a proper trilinear integration!
        float cell_t = mix(current_t, next_t, random());
        vec3 interpolation_alpha = (vm * cell_t + vn) - vec3(cell);
        float sigma_value = interpolated_sigma(object, interpolation_alpha, sigma_values);
        // homogeneous Transmittance
        float dt = next_t - current_t;
        float voxel_transmittance = exp(-dt * sigma_value);
        vec3 Le = exitance_radiance(object, x + cell_t * w, -w);

        vec3 dL_dLe = dL_dR * T * (1 - voxel_transmittance);
        exitance_radiance_bw(object, x + cell_t*w, -w, dL_dLe);

        R -= T * (1 - voxel_transmittance) * Le;
        T *= voxel_transmittance;

        float dL_dsigma = dot(dL_dR, (T*Le - fw_T*env - R)*dt);

        interpolated_sigma_bw(object, interpolation_alpha, dL_dsigma, sigma_grad_values);
        update_cell_gradients(object, cell, sigma_grad_values);

        ivec3 cell_inc = ivec3(
            alpha.x <= alpha.y && alpha.x <= alpha.z,
            alpha.x > alpha.y && alpha.y <= alpha.z,
            alpha.x > alpha.z && alpha.y > alpha.z);

        current_t = next_t;
        alpha += cell_inc * alpha_inc;
        cell += cell_inc * side;

//        if (T < 0.0001)
//        return;
    }

    /// T = exp(-tau);
    // dT_dtau = -exp(-tau)
}



//float DDA_TransmittanceEmission(inout uvec4 seed, vec3 x, vec3 w, float d, vec3 b_min, vec3 b_max, ivec3 dim, out vec3 emission)
//{
//    vec3 b_size = b_max - b_min;
//    vec3 cell_size = b_size / dim;
//    ivec3 cell = ivec3((x - b_min) * dim / b_size);
//    cell = clamp(cell, ivec3(0), dim - ivec3(1));
//    vec3 alpha_inc = cell_size / max(vec3(0.0000001), abs(w));
//	ivec3 side = ivec3(sign(w));
//	vec3 corner = (cell + side * 0.5 + vec3(0.5)) * cell_size + b_min;
//	vec3 alpha = abs(corner - x) / max(vec3(0.0000001), abs(w));
//    float T = 1;
//    emission = vec3(0);
//    float current_t = 0;
//    while(current_t < d - 0.00001){
//	    float next_t = min(d, min(alpha.x, min(alpha.y, alpha.z)));
//		ivec3 cell_inc = ivec3(
//			alpha.x <= alpha.y && alpha.x <= alpha.z,
//			alpha.x > alpha.y && alpha.y <= alpha.z,
//			alpha.x > alpha.z && alpha.y > alpha.z);
//        float dt = next_t - current_t;
//		float a = random(seed);
//        vec3 xt = x + (next_t*a + current_t*(1-a))*w;
//        float voxel_density = sample_sigma(xt);
//        float voxel_transmittance = exp(-dt * voxel_density * SIGMA_SCALE);
//        vec3 Le = sample_emission(xt, w);
//        emission += T * (1 - voxel_transmittance) * Le;
//        T *= voxel_transmittance;
//        if (T < 0.0000001)
//        return 0;
//		current_t = next_t;
//		alpha += cell_inc * alpha_inc;
//		cell += cell_inc * side;
//    }
//    return T;
//}
//
//
//void DDA_TransmittanceEmission_bw(inout uvec4 seed, vec3 x, vec3 w, float d, vec3 b_min, vec3 b_max, ivec3 dim, float fw_T, vec3 emission, float dL_dT, vec3 dL_demission)
//{
//    vec3 b_size = b_max - b_min;
//    vec3 cell_size = b_size / dim;
//    ivec3 cell = ivec3((x - b_min) * dim / b_size);
//    cell = clamp(cell, ivec3(0), dim - ivec3(1));
//    vec3 alpha_inc = cell_size / max(vec3(0.0000001), abs(w));
//	ivec3 side = ivec3(sign(w));
//	vec3 corner = (cell + side * 0.5 + vec3(0.5)) * cell_size + b_min;
//	vec3 alpha = abs(corner - x) / max(vec3(0.0000001), abs(w));
//    float T = 1;
//    float current_t = 0;
//    while(current_t < d - 0.00001){
//	    float next_t = min(d, min(alpha.x, min(alpha.y, alpha.z)));
//		ivec3 cell_inc = ivec3(
//			alpha.x <= alpha.y && alpha.x <= alpha.z,
//			alpha.x > alpha.y && alpha.y <= alpha.z,
//			alpha.x > alpha.z && alpha.y > alpha.z);
//		float a = random(seed);
//        float dt = next_t - current_t;
//        vec3 xt = x + (next_t*a + current_t*(1-a))*w;
//        float voxel_density = max(sample_sigma(xt), 0.0);
//
//        float voxel_transmittance = exp(-dt * voxel_density * SIGMA_SCALE);
//
//        float demission_de = T * (1 - voxel_transmittance);
//        vec3 Le = sample_emission(xt, w);
//        sample_emission_bw(xt, w, dL_demission * demission_de);
//        emission -= T * (1 - voxel_transmittance) * Le;
//
//        vec3 demission_dsigma1 = T * Le * voxel_transmittance * dt * SIGMA_SCALE;
//        vec3 demission_dsigma2 = -emission * dt * SIGMA_SCALE;
//
//        sample_sigma_bw(xt, - dL_dT * fw_T * SIGMA_SCALE * dt + dot(dL_demission, demission_dsigma1 + demission_dsigma2));
//
//        T *= voxel_transmittance;
//        if (T < 0.0000001)
//        return;
//        //tau += (next_t - current_t) * voxel_density * SIGMA_SCALE;
//		current_t = next_t;
//		alpha += cell_inc * alpha_inc;
//		cell += cell_inc * side;
//    }
//}                                                                                      