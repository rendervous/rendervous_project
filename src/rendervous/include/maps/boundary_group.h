FORWARD
{
    // input x, w,
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);

    rayQueryEXT rayQuery;
    rayQueryInitializeEXT(rayQuery,              // Ray query
                        accelerationStructureEXT(parameters.group_ads),                  // Top-level acceleration structure
                        gl_RayFlagsOpaqueEXT,  // Ray flags, here saying "treat all geometry as opaque"
                        0xFF,                  // 8-bit instance mask, here saying "trace against all instances"
                        x,                  // Ray origin
                        0.0,                   // Minimum t-value
                        w,            // Ray direction
                        10000.0);              // Maximum t-value

    // closest cached value among AABB candidates
    int patch_index = -1;
    float global_t = 100000.0;
    int from_outside = 0;

    while (rayQueryProceedEXT(rayQuery)) // traverse to find intersections
    {
        switch (rayQueryGetIntersectionTypeEXT(rayQuery, false))
        {
            case gl_RayQueryCandidateIntersectionTriangleEXT:
                // any triangle intersection is accepted
                rayQueryConfirmIntersectionEXT(rayQuery);
                // rayQueryTerminateEXT(rayQuery);
            break;
            #if ONLY_MESHES == 0
            case gl_RayQueryCandidateIntersectionAABBEXT:
                // any implicit hit is computed and cached in global surfel
                // Get the instance (patch) hit
                int index = 0;//rayQueryGetIntersectionInstanceIdEXT(rayQuery, false); // instance index
                mat4x3 w2o = rayQueryGetIntersectionWorldToObjectEXT(rayQuery, false);
                vec3 tx = transform_position(x, w2o);
                vec3 tw = transform_direction(w, w2o);
                float _local_input[6] = float[6](tx.x, tx.y, tx.z, tw.x, tw.y, tw.z);
                float _current[16];
                RaycastableInfo raycastInfo = RaycastableInfo(parameters.patches[index]);
                dynamic_forward(object, raycastInfo.callable_map, _local_input, _current);
                float current_t;
                int current_patch_index;
                Surfel current_surfel;
                if (hit2surfel(tx, tw, _current, current_t, current_patch_index, current_surfel) && (patch_index == -1 || current_t < global_t)) // replace closest surfel
                {
                    patch_index = index;//current_patch_index + int_ptr(parameters.patch_offsets).data[index];
                    global_t = current_t;
                    from_outside = dot(tw, current_surfel.N) < 0 ? 1 : 0;
                }
                rayQueryGenerateIntersectionEXT(rayQuery, current_t);
            break;
            #endif
        }
    }

    // check for the committed intersection to replace surfel if committed was a triangle
    if (rayQueryGetIntersectionTypeEXT(rayQuery, true) == gl_RayQueryCommittedIntersectionTriangleEXT)
    {
        from_outside = rayQueryGetIntersectionFrontFaceEXT(rayQuery, true) ? 1 : 0;
        patch_index = rayQueryGetIntersectionInstanceIdEXT(rayQuery, true); // instance index
        global_t = rayQueryGetIntersectionTEXT(rayQuery, true);
    }

    _output = float[3] (global_t, intBitsToFloat(from_outside), intBitsToFloat(patch_index));
}