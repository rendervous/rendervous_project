FORWARD
{
    // input x, w,
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);

    rayQueryEXT rayQuery_mesh;
    rayQueryInitializeEXT(rayQuery_mesh,              // Ray query
                        accelerationStructureEXT(parameters.mesh_ads),                  // Top-level acceleration structure
                        gl_RayFlagsOpaqueEXT,  // Ray flags, here saying "treat all geometry as opaque"
                        0xFF,                  // 8-bit instance mask, here saying "trace against all instances"
                        x,                  // Ray origin
                        0.0,                   // Minimum t-value
                        w,            // Ray direction
                        10000.0);              // Maximum t-value

    while(rayQueryProceedEXT(rayQuery_mesh))
        rayQueryConfirmIntersectionEXT(rayQuery_mesh);

    if (rayQueryGetIntersectionTypeEXT(rayQuery_mesh, true) != gl_RayQueryCommittedIntersectionNoneEXT)
    {
        int index = rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery_mesh, true);
        vec2 bar = rayQueryGetIntersectionBarycentricsEXT(rayQuery_mesh, true);
        float t = rayQueryGetIntersectionTEXT(rayQuery_mesh, true);
        surfel2array(t, 0, sample_surfel(MeshInfo(parameters.mesh_info), index, bar), _output);
    }
    else
        noHit2array(_output);
}