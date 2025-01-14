
void emitters(map_object, vec3 x, vec3 w, out vec3 E)
{
    float[3] _output;
    forward(parameters.emitters, float[6](x.x, x.y, x.w, w.x, w.y, w.z), _output);
    E = vec3(_output[0], _output[1], _output[2]);
}