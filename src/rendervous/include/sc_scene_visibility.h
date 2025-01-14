/*
Computes the radiance through a path of a scene with surface and volume scattering
Requires:
- surfaces
- volumes_scattering
*/

//float scene_visibility2(map_object, vec3 xa, vec3 xb)
//{
//    vec3 x = xa;
//    vec3 w = xb - xa;
//    float total_distance = length(w);
//    if (total_distance <= 0.0000001)
//        return 1.0;
//    w /= total_distance;
//
//    float T = 1.0;
//
//    for (int i=0; i<50; i++)
//    {
//        float d;
//        int patch_index;
//        bool is_entering;
//        if (!raycast(object, x, w, d, is_entering, patch_index))
//            return T;
//
//        VSPatchInfo patch_info = parameters.patch_info[patch_index];
//
//        GPUPtr current_medium = is_entering ? 0 : patch_info.inside_medium;
//
//        float vd = min(total_distance, d);
//
//        if (current_medium != 0)
//        {
//            float vT;
//            vec3 vW, vA;
//            vec3 xo, wo;
//            medium_traversal(object, current_medium, x, w, vd, vT, xo, wo, vW, vA);
//            T *= vT;
//            // T *= exp(-vd*5);
//        }
//
//        total_distance -= vd;
//
//        if (total_distance <= 0.00001) // traversed segment or opaque estimation within a volume
//        return T;
//
//        // Check scatter at surface
//        if (T < 0.000001 || patch_info.surface_scatters != 0)
//            return 0; // surface between xa-xb
//
//        x += w * (vd + 0.00001); // traverse volume distance and the epsilon to traspass the boundary in same direction
//        total_distance -= 0.00001;
//    }
//
//    return 0.0;
//}

void scene_visibility_bw(map_object, vec3 xa, vec3 xb, float out_T, float dL_dT)
{
    //PRINT("Inside Backward of visibility");

    if (out_T == 0.0)
    return; // no backprop necessary

    vec3 x = xa;
    vec3 w = xb - xa;
    float total_distance = length(w);
    if (total_distance <= 0.0000001)
        return;
    w /= total_distance;

    float T = 1.0;

    GPUPtr current_medium = 0; // assuming for now all rays starts outside all medium
    GPUPtr stacked_medium [4] = { 0, 0, 0, 0 };
    int medium_level = -1;

    for (int i=0; i<50; i++)
    {
        float d;
        int patch_index;
        bool is_entering;
        if (!raycast_from_bw(object, x, w, d, is_entering, patch_index))
            return;

        VSPatchInfo patch_info = parameters.patch_info[patch_index];

        if (medium_level == -1 && !is_entering)
        {
            medium_level = 0;
            current_medium = patch_info.inside_medium;
        }

        float vd = min(total_distance, d);

        if (current_medium != 0)
        {
            vec3 vW, vA;
            vec3 xo, wo;
            SAVE_SEED(before_traversal)
            medium_traversal(object, current_medium, x, w, vd, xo, wo, vW, vA);
            SAVE_SEED(after_traversal)
            float vT = (vW.x + vW.y + vW.z) / 3;
            T *= vT;
            vec3 dL_dW = vec3(dL_dT * out_T / vT)/3;
            dL_dW = mix(dL_dW, vec3(0.0), isnan(dL_dW));
            SET_SEED(before_traversal)
            medium_traversal_bw(object, current_medium, x, w, vd, dL_dW, vec3(0.0));
            // medium_traversal_bw(object, current_medium, x, w, vd, vW, vA, dL_dW, vec3(0.0));

            ASSERT(get_seed() == SEED(after_traversal), "sc_scene_visibility.h: Seed inconsistency during medium traversal")

            SET_SEED(after_traversal)
        }

        total_distance -= vd;

        if (total_distance <= 0.00001) // traversed segment or opaque estimation within a volume
        return;

        // Check scatter at surface
        if (T < 0.000001 || patch_info.surface_scatters != 0)
            return; // surface between xa-xb

        if (is_entering)
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

        x += w * (vd + 0.00001); // traverse volume distance and the epsilon to traspass the boundary in same direction
        total_distance -= 0.00001;
    }
}


float scene_visibility(map_object, vec3 xa, vec3 xb)
{
    vec3 x = xa;
    vec3 w = xb - xa;
    float total_distance = length(w);
    if (total_distance <= 0.0000001)
        return 1.0;
    w /= total_distance;

    float T = 1.0;

    GPUPtr current_medium = 0; // assuming for now all rays starts outside all medium
    GPUPtr stacked_medium [4] = { 0, 0, 0, 0 };
    int medium_level = -1;

    for (int i=0; i<50; i++)
    {
        float d;
        int patch_index;
        bool is_entering;
        if (!raycast(object, x, w, d, is_entering, patch_index))
            return T;

        VSPatchInfo patch_info = parameters.patch_info[patch_index];

        if (medium_level == -1 && !is_entering)
        {
            medium_level = 0;
            current_medium = patch_info.inside_medium;
        }

        float vd = min(total_distance, d);

        if (current_medium != 0)
        {
            vec3 vW, vA;
            vec3 xo, wo;
            medium_traversal(object, current_medium, x, w, vd, xo, wo, vW, vA);
            float vT = (wo == w ? 1.0 : 0.0) * (vW.x + vW.y + vW.z) / 3;
            T *= vT;
        }

        total_distance -= vd;

        if (total_distance <= 0.00001) // traversed segment or opaque estimation within a volume
        return T;

        // Check scatter at surface
        if (T < 0.000001) return 0.0;

        if (patch_info.surface_scatters != 0)
            return 0.0; // surface between xa-xb

        if (is_entering)
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

        x += w * (vd + 0.00001); // traverse volume distance and the epsilon to traspass the boundary in same direction
        total_distance -= 0.00001;
    }

    return 0.0;
}
