FORWARD
{
    vec4 p = vec4(_input[0], _input[1], _input[2], 1.0);
    p = parameters.projection * p;
    if (p.z < 0.0 || p.z > p.w || p.x < -p.w || p.x > p.w || p.y < -p.w || p.y > p.w) // clip
    {
        for (int i=0; i<OUTPUT_DIM; i++)
            _output[i] = 0.0;
        return;
    }
    p.xyz /= p.w;
    forward(parameters.field, float[3](p.x, p.y, p.z), _output);
}

BACKWARD
{
    vec4 p = vec4(_input[0], _input[1], _input[2], 1.0);
    p = parameters.projection * p;
    if (p.z < 0.0 || p.z > p.w || p.x < -p.w || p.x > p.w || p.y < -p.w || p.y > p.w) // clip
        return;
    p.xyz /= p.w;
    float inner_input_grad[3];
    backward(parameters.field, float[3](p.x, p.y, p.z), _output_grad, inner_input_grad);
}