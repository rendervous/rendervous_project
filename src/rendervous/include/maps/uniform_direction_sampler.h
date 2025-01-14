FORWARD
{
    vec3 w_out = randomDirection();
    _output = float[4](w_out.x, w_out.y, w_out.z, 4 * pi);
}