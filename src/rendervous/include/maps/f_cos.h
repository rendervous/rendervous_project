ACTIVATION_FUNCTION

float activation_fw(map_object, float x)
{
    return cos(x);
}

void activation_bw(map_object, float x, float dL_dy, inout float dL_dx)
{
    dL_dx -= dL_dy * sin(x);
}