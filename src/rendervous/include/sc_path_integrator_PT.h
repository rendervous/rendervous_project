/*
Computes the radiance through a path of a scene with surface and volume scattering
Requires:
- surfaces
- environment
- surface_scattering_sampler
- surface_emission
- volumes_scattering
*/

vec3 path_trace(map_object, vec3 x, vec3 w)
{
    vec3 W = vec3(1.0);
    vec3 A = vec3(0.0);

    for (int i=0; i<50; i++)
    {
        float d;
        int patch_index;
        Surfel surfel;
        if (!raycast(object, x, w, d, patch_index, surfel)) // skybox
            return A + W * environment(object, w);

        GPUPtr current_medium = 0;
        if (dot(surfel.N, w) < 0) /* hit from outside */
            current_medium = parameters.outside_media == 0 ? 0 : GPUPtr_ptr(parameters.outside_media).data[patch_index];
        else
            current_medium = parameters.inside_media == 0 ? 0 : GPUPtr_ptr(parameters.inside_media).data[patch_index];

        if (current_medium != 0)
        while(true) //for (int j=0; j<1000; j++)
        {
            float vT;
            vec3 vW, vA;
            vec3 xo, wo;
            volume_scattering(object, current_medium, x, w, d, vT, xo, wo, vW, vA);

            #ifdef MEDIA_FILTER
            #if MEDIA_FILTER == 0
            #else
            #if MEDIA_FILTER == 1  // only transmitted
            vW = vec3(0.0);
            vA = vec3(0.0);
            #else
            #if MEDIA_FILTER == 2  // only emitted
            vW = vec3(0.0);
            #else // only scattered
            vA = vec3(0.0);
            #endif
            #endif
            #endif
            #endif

            A += W * vA;
            float prW = (1 - vT) * (vW.x + vW.y + vW.z) / 3;
            float prT = vT;// / (vT + prW);

            if (random() < prT)
            {
                //W *= vT / prT;
                break;
            }

            W *= vW;// * (1 - vT) / (1 - prT);

            if (!raycast(object, xo - wo * 0.0001, wo, d, patch_index, surfel))
                return vec3(1, 0, 1);//A + W * environment(object, wo);

            d -= 0.0001;

            x = xo;
            w = wo;

//            if (j == 1000 - 1)
//            {
//                A = vec3(1.0, 0.0, 1.0);
//                W = vec3(0.0);
//                break;
//            }

//            if (all(equal(w, vec3(0.0))) || all(lessThan(W, vec3(0.000001)))) // collapsed direction or importance
//                return A;
        }

        // Accumulate emission
        vec3 E;
        GPUPtr emission_map = parameters.surface_emission == 0 ? 0 : GPUPtr_ptr(parameters.surface_emission).data[patch_index];
        if (emission_map != 0)
        {
            surface_emission(object, emission_map, w, surfel, E);
            A += W * E;
        }

        // Scatter at surface
        vec3 Ws;
        vec3 ws;
        float ws_pdf;
        GPUPtr scattering_sampler_map = parameters.surface_scattering_sampler == 0 ? 0 : GPUPtr_ptr(parameters.surface_scattering_sampler).data[patch_index];
        //vec3 facedNormal;
        if (scattering_sampler_map != 0)
        {
            sample_surface_scattering(object, scattering_sampler_map, w, surfel, ws, Ws, ws_pdf);
            //facedNormal = dot(ws, surfel.N) > 0 ? surfel.N : -surfel.N;
            w = ws; // change direction
            W *= Ws;
        }
        //else
            //facedNormal = dot(w, surfel.N) > 0 ? surfel.N : -surfel.N;
        x = surfel.P + w * 0.00001;// facedNormal * 0.001 + surfel.P; // move to surface point away to prevent self shadowing

        // check for early stops
        if (all(equal(w, vec3(0.0))) || all(lessThan(W, vec3(0.000001)))) // collapsed direction or importance
        break;
    }
    return A;
}
//
//vec3 path_trace_NEE(map_object, vec3 x, vec3 w)
//{
//    vec3 W = vec3(1.0);
//    vec3 A = vec3(0.0);
//
//    int_ptr outside_media = int_ptr(parameters.outside_media);
//    int_ptr inside_media = int_ptr(parameters.inside_media);
//
//    for (int i=0; i<20; i++)
//    {
//        HitInfo hit = raycast(object, x, w);
//        int patch_index = hit.patch_index;
//        if (patch_index == -1) // skybox
//        {
//            if (i == 0)
//                return environment(object, w);
//            return A;
//        }
//        float d = hit.t;
//        int current_medium_index = int_ptr((dot(hit.N, w) < 0) /* hit from outside */ ? parameters.outside_media : parameters.inside_media).data[patch_index];
//        if (current_medium_index != -1) // no empty medium
//        {
//            float vT;
//            vec3 vW, vA;
//            vec3 xo, wo;
//            volume_scattering(object, current_medium_index, x, w, d, vT, xo, wo, vW, vA);
//            A += W * vA;
//            float prW = (vW.x + vW.y + vW.z) / 3 + 0.0000001;
//            float prS = prW / (vT + prW);
//            if (random() < prS)
//            {
//                x = xo - wo * 0.0001;
//                w = wo;
//                hit = raycast(object, x, w);
//                W *= vW / prS;
//            }
//            else {
//                W *= vT / (1 - prS);
//            }
//        }
//        vec3 sW, sA;
//        vec3 sw;
//        // Scatter at surface
//        surface_scatter(object, x, w, hit, sw, sW, sA);
//        x += w * hit.t; // move to surface point
//        vec3 facedNormal = dot(sw, hit.N) > 0 ? hit.N : -hit.N;
//        x += facedNormal * 0.0001; // move away to prevent self shadowing
//        w = sw; // change direction
//        A += W * sA;
//        W *= sW;
//
//        // check for early stops
//        if (all(equal(w, vec3(0.0))) || all(lessThan(W, vec3(0.000001)))) // collapsed direction or importance
//        break;
//    }
//    return A;
//}