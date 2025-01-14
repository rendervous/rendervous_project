FORWARD
{
    vec2 c = vec2(_input[0], _input[1]);
    vec3 w = oct2dir(c);
    _output = float[3](w.x, w.y, w.z);
}