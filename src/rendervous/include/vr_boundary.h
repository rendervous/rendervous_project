bool boundary(map_object, vec3 x, vec3 w, out float tMin, out float tMax)
{
    float value[2];
    forward(parameters.boundary, float[6](x.x, x.y, x.z, w.x, w.y, w.z), value);
    tMin = value[0];
    tMax = value[1];
    tMin = max(0, tMin);
    if (tMax < tMin)
        return false;
    return true;
}