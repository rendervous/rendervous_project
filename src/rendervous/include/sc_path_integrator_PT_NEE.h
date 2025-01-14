/*
Computes the radiance through a path of a scene with surface and volume scattering
Requires:
- surfaces
- environment
- surface_emission
- emitters_sampler_pdf
- emitters_sampler
- surface_scattering
- surface_scattering_sampler
- volumes_scattering
*/

vec3 path_trace_NEE(map_object, vec3 x, vec3 w)
{
    vec3 W = vec3(1.0);
    vec3 A = vec3(0.0);

    // Add directly transmitted radiance from the explicit emitters
    vec3 E;
    emitters(object, x, w, E);
    A += E;

    // for each sub-path
    for (int i=0; i<15; i++)
    {
        int patch_index;
        float d;
        Surfel hit;
        if (!raycast(object, x, w, d, patch_index, hit))
            // Add final environment light not considered in explicit emitters
            return W * environment(object, w) + A;

        // If surface is hit, check the medium the path is traversing
        GPUPtr current_medium = 0;
        if (dot(hit.N, w) < 0) /* hit from outside */
            current_medium = parameters.outside_media == 0 ? 0 : GPUPtr_ptr(parameters.outside_media).data[patch_index];
        else
            current_medium = parameters.inside_media == 0 ? 0 : GPUPtr_ptr(parameters.inside_media).data[patch_index];

        // If medium is not null, traverse medium
        if (current_medium != 0)
        {
            float vT;
            vec3 vW, vA;
            vec3 xo, wo;
            volume_scattering(object, current_medium, x, w, d, vT, xo, wo, vW, vA);
            A += W * vA;
            float prW = (vW.x + vW.y + vW.z) / 3; // prob of scatter
            float prT = vT / (vT + prW); // prob of transmittance
            // Continue path either at scattered position or transmitted position
            if (random() >= prT)
            {
                x = xo - wo * 0.0001;
                w = wo;
                // recover surfel of scattered position
                raycast(object, x, w, d, patch_index, hit);
                W *= vW / (1 - prT);
            }
            else {
                W *= vT / prT;
            }`
        }

        // Add surface emission (that is not considered by emitters)
        vec3 E;
        GPUPtr emission_map = parameters.surface_emission == 0 ? 0 : GPUPtr_ptr(parameters.surface_emission).data[patch_index];
        if (emission_map != 0)
        {
            surface_emission(object, emission_map, w, hit, E);
            A += W * E;
        }

        // Sample scatter at surface
        vec3 Ws;
        vec3 ws;
        float pdfS_ws;
        GPUPtr scattering_sampler_map = parameters.surface_scattering_sampler == 0 ? 0 : GPUPtr_ptr(parameters.surface_scattering_sampler).data[patch_index];
        vec3 inFacedNormal = dot(w, hit.N) < 0 ? hit.N : -hit.N;
        if (scattering_sampler_map != 0) // some interface
        {
            sample_surface_scattering(object, scattering_sampler_map, w, hit, ws, Ws);
            vec3 outFacedNormal = dot(ws, hit.N) > 0 ? hit.N : -hit.N;
            // Get direct radiance in ws direction
            vec3 sE;
            emitters(object, hit.P, ws, sE);

            // Sample emitters
            vec3 neeE; vec3 we; float pdfE_we;
            emitters_sampler(object, hit.P, inFacedNormal, we, neeE, pdfE_we);
            vec3 We;
            GPUPtr scattering_map = GPUPtr_ptr(parameters.surface_scattering).data[hit.patch_index];
            surface_scattering(object, scattering_map, w, we, hit, We);

            // pdfE_ws
            float pdfE_ws;
            emitters_sampler_pdf(object, hit.P, inFacedNormal, ws, pdfE_ws);

            // pdfS_we
            float pdfS_we;
            GPUPtr scattering_sampler_pdf_map = GPUPtr_ptr(parameters.surface_scattering_sampler_pdf).data[hit.patch_index];
            surface_scattering_sampler_pdf(objcet, scattering_sampler_pdf_map, w, we, hit, pdfS_we);

            x = hit.P + outFacedNormal * 0.0001; // move away to prevent self shadowing
            w = sw; // change direction
            W *= sW;

            // Add NEE MIS (pdfE_we^b * BSDF(we) * E(we)/ (pdfE_we^b + pdfE_ws^b) + pdfS_ws^b * BSDF(ws) * E(ws) / (pdfS_ws^b + pdfS_we^b)
            float weight_we = pdfE_we == -1 ? 1.0 : pow(pdfE_we, 2) / (pow(pdfE_we, 2) + pow(pdfE_ws, 2));
            float weight_ws = pdfS_ws == -1 ? 1.0 : pow(pdfS_ws, 2) / (pow(pdfS_ws, 2) + pow(pdfS_we, 2));
            float mis_E = weight_we * We * neeE + weight_ws * Ws * sE;
            A += W * mis_E;
        }
        else
        {
        // move to the other side of the surface
            x = hit.P - inFacedNormal * 0.0001;
        }

        // check for early stops
        if (all(equal(w, vec3(0.0))) || all(lessThan(W, vec3(0.000001)))) // collapsed direction or importance
        break;
    }
    return A;
}
