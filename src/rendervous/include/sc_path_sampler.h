/*
Samples a path with weight through a scene with surface and volume scattering
Requires:
- surfaces
- surface_scattering_sampler
- volumes_scattering
*/


void path_sampler(map_object, vec3 x, vec3 win, out vec3 w, out vec3 W)
{
    W = vec3(1.0);
    w = win;
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
            return;

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
        x = surfel.P + surfel.G * 0.0001 * sign(dot(surfel.G, w));
    }
}
