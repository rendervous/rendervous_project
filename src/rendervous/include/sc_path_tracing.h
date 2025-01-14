/*
Computes the radiance through a path of a scene with surface and volume scattering
Requires:
- surfaces
- environment
- surface_scattering_sampler
- surface_gathering
- volumes_scattering
*/


void path_trace(map_object, vec3 x, vec3 w, out vec3 A)
{
    // Start A with direct radiance
    float[3] dr;
    forward(parameters.direct_gathering, float[6](x.x, x.y, x.z, w.x, w.y, w.z), dr);
    A = vec3(dr[0], dr[1], dr[2]);
    //A = vec3(0.0);
    vec3 W = vec3(1.0);
    GPUPtr current_medium = 0; // assuming for now all rays starts outside all medium
    GPUPtr stacked_medium [4] = { 0, 0, 0, 0 };
    int medium_level = -1;

    int bounces = 0;
    while(true)
    {
        if (all(lessThan(W, vec3(0.0000001)))) // absorption
            return;

        float d;
        int patch_index;
        Surfel surfel;
        if (!raycast(object, x, w, d, patch_index, surfel)) // skybox
        {
            vec3 E;
            environment(object, x, w, E);
            A += W * E;
            return;
        }

        PTPatchInfo patch_info = parameters.patch_info[patch_index];
        bool from_inside = dot(w, surfel.G) > 0;

        if (medium_level == -1 && from_inside) // started already inside a medium
        // TODO: This condition can be removed with a proper initial medium stack calculation
        {
            medium_level = 0;
            current_medium = patch_info.inside_medium;
        }

        if (current_medium != 0)
        {
            vec3 xo, wo;
            vec3 vW, vA;
            medium_traversal(object, current_medium, x, w, d, xo, wo, vW, vA);
            A += W * vA;
            W *= vW;
            if (wo != w) // scatters inside the medium
            {
                x = xo;
                w = wo;
                continue;
            }
        }

        // else is a transmitted path, compute surface bounce
        // Bounce at a surface
        bounces ++;

        // Gather radiance at surfaces
        vec3 sA;
        GPUPtr gathering_map = patch_info.surface_gathering;
        if (gathering_map != 0)
        {
            surface_gathering(object, gathering_map, w, surfel, sA);
            A += W * sA;
        }

        // Scatter at surface
        vec3 Ws = vec3(1.0);
        vec3 ws = w;
        float ws_pdf = -1;
        GPUPtr scattering_sampler_map = patch_info.surface_scattering_sampler;
        if (scattering_sampler_map != 0)
            surface_scattering_sampler(object, scattering_sampler_map, w, surfel, ws, Ws, ws_pdf);

        bool to_inside = dot(ws, surfel.G) < 0;
        if (to_inside != from_inside) // traversing
            if (to_inside)
            { // stack current medium and change for patch_info
                if (medium_level >= 0)
                    stacked_medium[medium_level] = current_medium;
                if (medium_level < 3)
                    medium_level ++;
                current_medium = patch_info.inside_medium;
            }
            else
            { // pop current medium from stack
                medium_level--;
                if (medium_level == -1)
                    current_medium = 0;
                else
                    current_medium = stacked_medium[medium_level];
            }

        W *= Ws;
        w = ws; // change direction

        if (ws_pdf > 0 && bounces > 2) // not delta and after 2 bounces, apply Russian roulette
        {
            if (random() < 0.5)
            {
                W = vec3(0.0);
                break;
            }
            W *= 2;
        }
        x = surfel.P + w * 0.0001;
    }
}


void path_trace_fw(map_object, vec3 x, vec3 w, out vec3 A)
{
    // Start A with direct radiance
    BEGIN_BRANCH(DG)
    float[3] dr;
    forward(parameters.direct_gathering, float[6](x.x, x.y, x.z, w.x, w.y, w.z), dr);
    A = vec3(dr[0], dr[1], dr[2]);
    END_BRANCH(DG)
    //A = vec3(0.0);
    vec3 W = vec3(1.0);
    GPUPtr current_medium = 0; // assuming for now all rays starts outside all medium
    GPUPtr stacked_medium [4] = { 0, 0, 0, 0 };
    int medium_level = -1;

    int bounces = 0;
    while(true)
    {
        if (all(lessThan(W, vec3(0.0000001)))) // absorption
            return;

        float d;
        int patch_index;
        Surfel surfel;
        if (!raycast(object, x, w, d, patch_index, surfel)) // skybox
        {
            vec3 E;
            environment(object, x, w, E);
            A += W * E;
            return;
        }

        PTPatchInfo patch_info = parameters.patch_info[patch_index];
        bool from_inside = dot(w, surfel.G) > 0;

        if (medium_level == -1 && from_inside) // started already inside a medium
        // TODO: This condition can be removed with a proper initial medium stack calculation
        {
            medium_level = 0;
            current_medium = patch_info.inside_medium;
        }

        if (current_medium != 0)
        {
            vec3 xo, wo;
            vec3 vW, vA;
            // Make a branch to allow different seed-sequences for fw and bw when replaying medium traversal
            BEGIN_BRANCH(MT);
            medium_traversal(object, current_medium, x, w, d, xo, wo, vW, vA);
            END_BRANCH(MT);
            A += W * vA;
            W *= vW;
            if (wo != w) // scatters inside the medium
            {
                x = xo;
                w = wo;
                continue;
            }
        }

        // else is a transmitted path, compute surface bounce
        // Bounce at a surface
        bounces ++;

        // Gather radiance at surfaces
        vec3 sA;
        GPUPtr gathering_map = patch_info.surface_gathering;
        if (gathering_map != 0)
        {
            surface_gathering(object, gathering_map, w, surfel, sA);
            A += W * sA;
        }

        // Scatter at surface
        vec3 Ws = vec3(1.0);
        vec3 ws = w;
        float ws_pdf = -1;
        GPUPtr scattering_sampler_map = patch_info.surface_scattering_sampler;
        if (scattering_sampler_map != 0)
            surface_scattering_sampler(object, scattering_sampler_map, w, surfel, ws, Ws, ws_pdf);

        bool to_inside = dot(ws, surfel.G) < 0;
        if (to_inside != from_inside) // traversing
            if (to_inside)
            { // stack current medium and change for patch_info
                if (medium_level >= 0)
                    stacked_medium[medium_level] = current_medium;
                if (medium_level < 3)
                    medium_level ++;
                current_medium = patch_info.inside_medium;
            }
            else
            { // pop current medium from stack
                medium_level--;
                if (medium_level == -1)
                    current_medium = 0;
                else
                    current_medium = stacked_medium[medium_level];
            }

        W *= Ws;
        w = ws; // change direction

        if (ws_pdf > 0 && bounces > 2) // not delta and after 2 bounces, apply Russian roulette
        {
            if (random() < 0.5)
            {
                W = vec3(0.0);
                break;
            }
            W *= 2;
        }
        x = surfel.P + w * 0.0001;
    }
}

void path_trace_bw(map_object, vec3 x, vec3 w, vec3 out_A, vec3 dL_dA)
{
    // Tracker for the replay
    BEGIN_BRANCH(DG)
    SAVE_SEED(before_direct_gathering)
    float[3] dr;
    forward(parameters.direct_gathering, float[6](x.x, x.y, x.z, w.x, w.y, w.z), dr);
    SET_SEED(before_direct_gathering)
    float[6] _input_grad;
    backward(parameters.direct_gathering, float[6](x.x, x.y, x.z, w.x, w.y, w.z), dr, float[3](dL_dA.x, dL_dA.y, dL_dA.z), _input_grad);
    END_BRANCH(DG)
    // A = vec3(dr[0], dr[1], dr[2]);
    // A = vec3(0.0);
    vec3 rem_A = out_A - vec3(dr[0], dr[1], dr[2]);
    vec3 W = vec3(1.0);
    GPUPtr current_medium = 0; // assuming for now all rays starts outside all medium
    GPUPtr stacked_medium [4] = { 0, 0, 0, 0 };
    int medium_level = -1;

    int bounces = 0;
    while(true)
    {
        if (all(lessThan(W, vec3(0.0000001)))) // absorption
            return;

        float d;
        int patch_index;
        Surfel surfel;
        if (!raycast(object, x, w, d, patch_index, surfel)) // skybox
        {
            // vec3 E;
            vec3 dL_dE = dL_dA * W;
            environment_bw(object, x, w, dL_dE);
            // A += W * E;
            // rem_A -= W*E; // not necessary, should be 0 here at the last term
            return;
        }

        PTPatchInfo patch_info = parameters.patch_info[patch_index];
        bool from_inside = dot(w, surfel.G) > 0;

        if (medium_level == -1 && from_inside) // started already inside a medium
        // TODO: This condition can be removed with a proper initial medium stack calculation
        {
            medium_level = 0;
            current_medium = patch_info.inside_medium;
        }

        if (current_medium != 0)
        {
            vec3 xo, wo;
            vec3 vW, vA;
            SAVE_SEED(before_medium_traversal);
            medium_traversal(object, current_medium, x, w, d, xo, wo, vW, vA);
            SAVE_SEED(after_medium_traversal);
            // A += W * vA;
            rem_A -= W * vA;
            vec3 dL_dvA = dL_dA * W;
            vec3 dL_dvW = dL_dA * rem_A / vW;
            dL_dvW = mix(dL_dvW, vec3(0.0), isnan(dL_dvW));
            SET_SEED(before_medium_traversal);
            medium_traversal_bw(object, current_medium, x, w, d, dL_dvW, dL_dvA);
            //medium_traversal_bw(object, current_medium, x, w, d, vW, vA, dL_dvW, dL_dvA);
            ASSERT(get_seed() == SEED(after_medium_traversal), "sc_path_tracing.h: Seed inconsistency during medium traversal")
            //SET_SEED(after_medium_traversal);
            W *= vW;
            if (wo != w) // scatters inside the medium
            {
                x = xo;
                w = wo;
                continue;
            }
        }

        // else is a transmitted path, compute surface bounce
        // Bounce at a surface
        bounces ++;

        // Gather radiance at surfaces
        vec3 sA;
        GPUPtr gathering_map = patch_info.surface_gathering;
        if (gathering_map != 0)
        {
            SAVE_SEED(before_gathering);
            surface_gathering(object, gathering_map, w, surfel, sA);
            //SAVE_SEED(after_gathering);
            SET_SEED(before_gathering);
            vec3 dL_dsA = dL_dA * W;
            surface_gathering_bw(object, gathering_map, w, surfel, sA, dL_dsA);
            //SET_SEED(after_gathering);
            // A += W * sA;
            rem_A -= W * sA;
        }

        // Scatter at surface
        // TODO: Implement backprop to scattering sampler
        vec3 Ws = vec3(1.0);
        vec3 ws = w;
        float ws_pdf = -1;
        GPUPtr scattering_sampler_map = patch_info.surface_scattering_sampler;
        if (scattering_sampler_map != 0)
            surface_scattering_sampler(object, scattering_sampler_map, w, surfel, ws, Ws, ws_pdf);

        bool to_inside = dot(ws, surfel.G) < 0;
        if (to_inside != from_inside) // traversing
            if (to_inside)
            { // stack current medium and change for patch_info
                if (medium_level >= 0)
                    stacked_medium[medium_level] = current_medium;
                if (medium_level < 3)
                    medium_level ++;
                current_medium = patch_info.inside_medium;
            }
            else
            { // pop current medium from stack
                medium_level--;
                if (medium_level == -1)
                    current_medium = 0;
                else
                    current_medium = stacked_medium[medium_level];
            }

        W *= Ws;
        w = ws; // change direction

        if (ws_pdf > 0 && bounces > 2) // not delta and after 2 bounces, apply Russian roulette
        {
            if (random() < 0.5)
            {
                W = vec3(0.0);
                break;
            }
            W *= 2;
        }
        x = surfel.P + w * 0.0001;
    }
}






//vec3 path_trace_2(map_object, vec3 x, vec3 w)
//{
//    vec3 W = vec3(1.0);
//    vec3 A = vec3(0.0);
//
//    int bounces = 0;
//    [[unroll]]
//    while(bounces < 50)
//    {
//        float d;
//        int patch_index;
//        bool from_outside;
//        if (!raycast(object, x, w, d, from_outside, patch_index)) // skybox
//        {
//            vec3 E;
//            environment(object, x, w, E);
//            return A + W * E;
//        }
//
//        PTPatchInfo patch_info = parameters.patch_info[patch_index];
//
//        GPUPtr current_medium = from_outside ? 0 : patch_info.inside_medium;
//
//        if (current_medium != 0)
//        {
//            float vT = 1.0;
//            vec3 vW = vec3(1.0);
//            vec3 xo, wo;
//            vec3 vA;
//            medium_traversal(object, current_medium, x, w, d, vT, xo, wo, vW, vA);
//            A += W * vA;
//
//            float Ps = (1 - vT)*(vW.x + vW.y + vW.z)/3;
//
//            if (random() < Ps) // volume scattering
//            {
//                W *= vW;
//                if (all(lessThan(W, vec3(0.000001)))) // collapsed direction or importance
//                    return A;
//                x = xo;
//                w = wo;
//                continue;
//            }
//
//            W *= vT / (1 - Ps);
//        }
//
//        // Bounce at a surface
//        bounces ++;
//
//        GPUPtr scattering_sampler_map = patch_info.surface_scattering_sampler;
//
//        x += w * d; // move to the surface
//
//        [[branch]]
//        if (scattering_sampler_map != 0)
//        {
//            Surfel surfel;
//            // repeat raycast to retrieve surfel
//            raycast(object, x - w*0.0001, w, d, patch_index, surfel);
//            if (patch_index == -1)
//            return A;
//
//            // Gather radiance at surfaces
//            vec3 sA;
//            GPUPtr gathering_map = patch_info.surface_gathering;
//            if (gathering_map != 0)
//            {
//                surface_gathering(object, gathering_map, w, surfel, sA);
//                A += W * sA;
//            }
//
//            // Scatter at surface
//            vec3 Ws;
//            vec3 ws;
//            float ws_pdf = -1;
//            sample_surface_scattering(object, scattering_sampler_map, w, surfel, ws, Ws, ws_pdf);
//
//            W *= Ws;
//            w = ws; // change direction
//        }
//
//        x += w * 0.0001; // move away an epsilon from the surface
//    }
//    return vec3(0.0);//vec3(1000000.0, 0, 1000000.0);
//}
//
//vec3 path_trace_b(map_object, vec3 x, vec3 w)
//{
//    vec3 W = vec3(1.0);
//    vec3 A = vec3(0.0);
//
//    int bounces = 0;
//    while(bounces < 50)
//    {
//        float d;
//        int patch_index;
//        Surfel surfel;
//        if (!raycast(object, x, w, d, patch_index, surfel)) // skybox
//        {
//            vec3 E;
//            environment(object, x, w, E);
//            return A + W * E;
//        }
//
//        PTPatchInfo patch_info = parameters.patch_info[patch_index];
//        bool from_inside = dot(w, surfel.G) > 0;
//
//        GPUPtr current_medium = from_inside ? patch_info.inside_medium : 0;
//
//        float vT = 1.0;
//        vec3 vW = vec3(1.0);
//        vec3 xo, wo;
//
//        if (current_medium != 0)
//        {
//            vec3 vA;
//            medium_traversal(object, current_medium, x, w, d, vT, xo, wo, vW, vA);
//            A += W * vA;
//        }
//
//        if (random() < vT)
//        {
//            // Bounce at a surface
//            bounces ++;
//
//            // Gather radiance at surfaces
//            vec3 sA;
//            GPUPtr gathering_map = patch_info.surface_gathering;
//            if (gathering_map != 0)
//            {
//                surface_gathering(object, gathering_map, w, surfel, sA);
//                A += W * sA;
//            }
//
//            // Scatter at surface
//            vec3 Ws = vec3(1.0);
//            vec3 ws = w;
//            float ws_pdf = -1;
//            GPUPtr scattering_sampler_map = patch_info.surface_scattering_sampler;
//            if (scattering_sampler_map != 0)
//                sample_surface_scattering(object, scattering_sampler_map, w, surfel, ws, Ws, ws_pdf);
//
//            W *= Ws;
//            w = ws; // change direction
//
////            if (ws_pdf > 0) // not delta
////            {
////                if (random() < 0.5)
////                return A;
////                W *= 2;
////            }
//
//            x = surfel.P + w * 0.0001;
//        }
//        else {
//            W *= vW;
//            if (all(lessThan(W, vec3(0.000001)))) // collapsed direction or importance
//                return A;
//            x = xo;
//            w = wo;
//        }
//    }
//    return vec3(0.0);//vec3(1000000.0, 0, 1000000.0);
//}
//
//
//
//
//
//
//
//vec3 path_trace2(map_object, vec3 x, vec3 w)
//{
//    vec3 W = vec3(1.0);
//    vec3 A = vec3(0.0);
//
//    [[unroll]]
//    for (int i=0; i<50; i++)
//    {
//        float d;
//        int patch_index;
//        Surfel surfel;
//        if (!raycast(object, x, w, d, patch_index, surfel)) // skybox
//        {
//            vec3 E;
//            environment(object, x, w, E);
//            return A + W * E;
//        }
//
////        return surfel.N;
//
//        PTPatchInfo patch_info = parameters.patch_info[patch_index];
//
//        GPUPtr current_medium = 0;
//        if (dot(surfel.G, w) < 0) /* hit from outside */
//            current_medium = patch_info.outside_medium;
//        else
//            current_medium = patch_info.inside_medium;
//
//        bool some_scatter = false;
//        //if (current_medium != 0)
//        while(true)
//        {
//            if (current_medium == 0)
//            break;
//
//            float vT;
//            vec3 vW, vA;
//            vec3 xo, wo;
//            medium_traversal(object, current_medium, x, w, d, vT, xo, wo, vW, vA);
//
//            A += W * vA;
//
//            if (random() < vT)
//                break;
//
//            some_scatter = true;
//
//            W *= vW;
//
//            if (wo == vec3(0.0) || all(lessThan(W, vec3(0.000001)))) // collapsed direction or importance
//                return A;
//
//            // Raycast to retrieve new distance towards scattered direction
//            bool is_entering;
//            // if (!raycast(object, xo - wo*0.000001, wo, d, is_entering, patch_index))
//            if (!raycast(object, xo - wo*0.0001, wo, d, is_entering, patch_index))
//                return A; //vec3(100.0, 0.0, .2);// + W * environment(object, wo);
//
//            d -= 0.0001;
//
//            x = xo;
//            w = wo;
//        }
//
//        // Raycast to retrieve the exit surfel only if some scatter
//        if (some_scatter && !raycast(object, x - w * 0.0001, w, d, patch_index, surfel))
//            // should always hit the surface, instruction below should never occur.
//            return A;// + W * environment(object, wo);
//
//        // update patch info if volume was traversed
//        patch_info = parameters.patch_info[patch_index];
//
//        // Gather radiance at surfaces
//        vec3 sA;
//        GPUPtr gathering_map = patch_info.surface_gathering;
//        if (gathering_map != 0)
//        {
//            surface_gathering(object, gathering_map, w, surfel, sA);
//            A += W * sA;
//        }
//
//        // Scatter at surface
//        vec3 Ws;
//        vec3 ws;
//        float ws_pdf= -1;
//        GPUPtr scattering_sampler_map = patch_info.surface_scattering_sampler;
//
//        if (scattering_sampler_map != 0)
//        {
//            sample_surface_scattering(object, scattering_sampler_map, w, surfel, ws, Ws, ws_pdf);
//            w = ws; // change direction
//            W *= Ws;
////            float s;
////            w = randomHSDirectionCosineWeighted(dot(w, surfel.N) < 0 ? surfel.N : -surfel.N, s);
////            W *= vec3(0.1,0.1,0.1);
//        }
//
//        if (ws_pdf > 0) // not delta
//        {
//            if (random() < 0.5)
//            return A;
//            W *= 2;
//        }
//
//        x = surfel.P + w * 0.0001;
//
//        // check for early stops after surface scattering
//        if (w == vec3(0.0) || all(lessThan(W, vec3(0.000001)))) // collapsed direction or importance
//        return A;
//    }
//    return vec3(0.0);//vec3(1000000.0, 0, 1000000.0);
//}
