float majorant(map_object, vec3 x, vec3 w, out float d)
{
    float value[2];
    forward(parameters.majorant, float[6](x.x, x.y, x.z, w.x, w.y, w.z), value);
    d = value[1];
    return value[0];
}