FORWARD
{
#if INPUT_DIM == 3
    vec3 w = vec3(_input[0], _input[1], _input[2]);
#else
    vec3 w = vec3(_input[0], _input[1], _input[2]);
#endif
    vec2 c = dir2oct(w);
    _output = float[2](c.x, c.y);
}