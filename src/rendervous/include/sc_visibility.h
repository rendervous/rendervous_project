

float visibility(map_object, vec3 xa, vec3 xb)
{
    float _output[1];
    forward(parameters.visibility, float[6](xa.x, xa.y, xa.z, xb.x, xb.y, xb.z), _output);
    return _output[0];
}

float ray_visibility(map_object, vec3 x, vec3 w)
{
    vec3 xb = w * 10000 + x;
    vec3 xa = x;
    float _output[1];
    forward(parameters.visibility, float[6](xa.x, xa.y, xa.z, xb.x, xb.y, xb.z), _output);
    return _output[0];
}

void ray_visibility_bw(map_object, vec3 x, vec3 w, float T, float dL_dT)
{
    vec3 xb = w * 10000 + x;
    vec3 xa = x;
    float _output[1] = float[1](T);
    float _output_grad[1] = float[1](dL_dT);
    float _input_grad[6];
    backward(parameters.visibility, float[6](xa.x, xa.y, xa.z, xb.x, xb.y, xb.z), _output, _output_grad, _input_grad);
}