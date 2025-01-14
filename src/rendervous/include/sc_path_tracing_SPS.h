/*
Computes the radiance through a path of a scene with surface and volume scattering
Requires:
- surfaces
- environment
- surface_scattering_sampler
- surface_gathering
- volumes_scattering
*/


void path_trace(map_object, vec3 x, vec3 w, out vec3 A, out vec3 after_A)
{
    A = vec3(0.0);
    after_A = vec3(0.0); // accumulation after singular vertex
    vec3 W = vec3(1.0); // Path throughput without first singular factor
    vec3 singular_W = vec3(1.0); // singular factor in the paththroughput

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
            A += W * singular_W * E;
            after_A += W * (1 - singular_W) * E;
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
            A += W * singular_W * vA;
            after_A += W * (1 - singular_W) * vA;

            vec3 current_singular = vec3(equal(vW, vec3(0.0)));
            W *= mix(vW, vec3(1.0), current_singular * singular_W);
            singular_W *= (1 - current_singular);

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
            A += W * singular_W * sA;
            after_A += W * (1 - singular_W) * sA;
        }

        // Scatter at surface
        vec3 ws = w;
        float ws_pdf = -1;
        GPUPtr scattering_sampler_map = patch_info.surface_scattering_sampler;
        if (scattering_sampler_map != 0)
        {
            vec3 sW = vec3(1.0);
            surface_scattering_sampler(object, scattering_sampler_map, w, surfel, ws, sW, ws_pdf);
            vec3 current_singular = vec3(equal(sW, vec3(0.0)));
            W *= mix(sW, vec3(1.0), current_singular * singular_W);
            singular_W *= (1 - current_singular);
        }

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


void path_trace_bw(map_object, vec3 x, vec3 w, vec3 out_A, vec3 out_after_A, vec3 dL_dA)
{
    // Tracker for the replay
    // A = vec3(0.0);
    vec3 rem_A = out_A;
    vec3 rem_after_A = out_after_A;
    vec3 W = vec3(1.0);
    vec3 singular_W = vec3(1.0);

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
            //SAVE_SEED(after_medium_traversal);
            // A += W * singular_W * vA;
            // after_A += W * (1 - singular_W) * vA;
            rem_A -= W * singular_W * vA;
            rem_after_A -= W * (1 - singular_W) * vA;

            vec3 current_singular = vec3(equal(vW, vec3(0.0)));

            vec3 dL_dvA = dL_dA * W * singular_W;
            vec3 dL_dvW = dL_dA * mix(rem_A * singular_W, rem_after_A, current_singular) / vW;
            dL_dvW = mix(dL_dvW, vec3(0.0), isnan(dL_dvW));

            SET_SEED(before_medium_traversal);
            medium_traversal_bw(object, current_medium, x, w, d, vW, vA, dL_dvW, dL_dvA);
            //ASSERT(get_seed() == SEED(after_medium_traversal), "sc_path_tracing.h: Seed inconsistency during medium traversal")
            //SET_SEED(after_medium_traversal);
            W *= mix(vW, vec3(1.0), current_singular * singular_W);
            singular_W *= (1 - current_singular);

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
            // A += W * singular_W * sA;
            // after_A += W * (1 - singular_W) * sA;
            rem_A -= W * singular_W * sA;
            rem_after_A -= W * (1 - singular_W) * sA;
        }

        // Scatter at surface
        // TODO: Implement backprop to scattering sampler
        vec3 ws = w;
        float ws_pdf = -1;
        GPUPtr scattering_sampler_map = patch_info.surface_scattering_sampler;
        if (scattering_sampler_map != 0)
        {
            vec3 sW = vec3(1.0);
            surface_scattering_sampler(object, scattering_sampler_map, w, surfel, ws, sW, ws_pdf);
            vec3 current_singular = vec3(equal(sW, vec3(0.0)));
            // TODO: surface scattering backprop!
            // ...
            W *= mix(sW, vec3(1.0), current_singular * singular_W);
            singular_W *= (1 - current_singular);
        }

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



