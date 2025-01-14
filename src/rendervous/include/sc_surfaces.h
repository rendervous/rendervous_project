/*
Represents objects with a scene that can be raycast
*/

bool raycast(map_object, vec3 x, vec3 w, out float t, out int patch_index, out Surfel surfel) {
    if (uint64_t(parameters.surfaces.data) == 0)
        return false;
    float _hit_info[16];
    forward(parameters.surfaces, float[6](x.x, x.y, x.z, w.x, w.y, w.z), _hit_info);
    return hit2surfel(x, w, _hit_info, t, patch_index, surfel);
}