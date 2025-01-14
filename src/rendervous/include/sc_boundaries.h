/*
Represents objects with a boundaries that can be raycast
*/

bool raycast(map_object, vec3 x, vec3 w, out float t) {
    if (uint64_t(parameters.boundaries.data) == 0)
        return false;
    float _hit_info[3];
    forward(parameters.boundaries, float[6](x.x, x.y, x.z, w.x, w.y, w.z), _hit_info);
    t = _hit_info[0];
    return floatBitsToInt(_hit_info[2]) >= 0;
}


bool raycast(map_object, vec3 x, vec3 w, out float t, out bool is_entering, out int patch_index) {
    if (uint64_t(parameters.boundaries.data) == 0)
        return false;
    float _hit_info[3];
    forward(parameters.boundaries, float[6](x.x, x.y, x.z, w.x, w.y, w.z), _hit_info);
    t = _hit_info[0];
    is_entering = floatBitsToInt(_hit_info[1]) == 1;
    patch_index = floatBitsToInt(_hit_info[2]);
    return patch_index >= 0;
}

bool raycast_from_bw(map_object, vec3 x, vec3 w, out float t, out bool is_entering, out int patch_index) {
    if (uint64_t(parameters.boundaries.data) == 0)
        return false;
    float _hit_info[3];
    forward(parameters.boundaries, float[6](x.x, x.y, x.z, w.x, w.y, w.z), _hit_info);
    t = _hit_info[0];
    is_entering = floatBitsToInt(_hit_info[1]) == 1;
    patch_index = floatBitsToInt(_hit_info[2]);
    return patch_index >= 0;
}