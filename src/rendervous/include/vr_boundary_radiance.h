vec3 boundary_radiance(map_object, vec3 x, vec3 w)
{
    float value[3];
    forward (parameters.boundary_radiance, float[](x.x, x.y, x.z, w.x, w.y, w.z), value);
    return vec3(value[0], value[1], value[2]);
}

void boundary_radiance_bw(map_object, vec3 x, vec3 w, vec3 dL_dbr)
{
    backward (parameters.boundary_radiance, float[](x.x, x.y, x.z, w.x, w.y, w.z), float[] (dL_dbr.x, dL_dbr.y, dL_dbr.z));
}