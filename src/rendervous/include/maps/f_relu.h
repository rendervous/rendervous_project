ACTIVATION_FUNCTION

float activation_fw(map_object, float x)
{
    return max(0, x);
}

void activation_bw(map_object, float x, float dL_dy, inout float dL_dx)
{
    if (x > 0) dL_dx += dL_dy;
}