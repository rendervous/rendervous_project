// Samples a path exit with respect to some strategy and return the W and A accumulation
void medium_path_integrator(map_object, vec3 x, vec3 w, float d, out vec3 xo, out vec3 wo, out vec3 W, out vec3 A);

// Backprop the gradients dL_dW and dL_dA (most likely using Path-replay)
void medium_path_integrator_bw(map_object, vec3 x, vec3 w, float d, vec3 dL_dW, vec3 dL_dA);

vec3 gathering(map_object, vec3 xs, vec3 ws)
{
    float _output[3];
    forward(parameters.gathering, float[6](xs.x, xs.y, xs.z, ws.x, ws.y, ws.z), _output);
    return vec3(_output[0], _output[1], _output[2]);
}

void gathering_bw(map_object, vec3 xs, vec3 ws, vec3 A, vec3 dL_dA)
{
    float[6] _input_grad;
    backward(parameters.gathering, float[6](xs.x, xs.y, xs.z, ws.x, ws.y, ws.z), float[3](A.x, A.y, A.z), float[3](dL_dA.x, dL_dA.y, dL_dA.z), _input_grad);
}

FORWARD {
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    float d = _input[6];
    vec3 xo, wo;
    vec3 W, A;
    medium_path_integrator(object, x, w, d, xo, wo, W, A);
    _output = float[12](
        xo.x, xo.y, xo.z,
        wo.x, wo.y, wo.z,
        W.x, W.y, W.z,
        A.x, A.y, A.z
    );
}

BACKWARD {
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    float d = _input[6];
    vec3 dL_dW = vec3(_output_grad[6], _output_grad[7], _output_grad[8]);
    vec3 dL_dA = vec3(_output_grad[9], _output_grad[10], _output_grad[11]);
    // backprop
    medium_path_integrator_bw(
        object, x, w, d, dL_dW, dL_dA
    );
    // TODO: no input update here! check how can be done...
}
