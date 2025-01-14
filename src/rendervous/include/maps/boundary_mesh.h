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

    int index = -1;
    int from_outside = 0;
    float t = 100000.0;

    if (rayQueryGetIntersectionTypeEXT(rayQuery_mesh, true) != gl_RayQueryCommittedIntersectionNoneEXT)
    {

        from_outside = rayQueryGetIntersectionFrontFaceEXT(rayQuery_mesh, true) ? 1 : 0;
        index = 0;
        t = rayQueryGetIntersectionTEXT(rayQuery_mesh, true);
    }
    _output = float[3](t, intBitsToFloat(from_outside), intBitsToFloat(index));
}