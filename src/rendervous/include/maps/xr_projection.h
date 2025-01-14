FORWARD
{
#if INPUT_DIM == 3
    vec3 w = vec3(_input[0], _input[1], _input[2]);
#else
    vec3 w = vec3(_input[3], _input[4], _input[5]);
#endif
    vec2 c = dir2xr(w);
    _output = float[2](c.x, c.y);
}