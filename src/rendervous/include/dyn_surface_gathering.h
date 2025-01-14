/*
Represents objects that can scatter in surfaces.
*/

void surface_gathering(map_object, GPUPtr gathering_map, vec3 w, Surfel hit, out vec3 A)
{
    float _input[20] = float[20](
        w.x, w.y, w.z,
        hit.P.x, hit.P.y, hit.P.z,
        hit.N.x, hit.N.y, hit.N.z,
        hit.G.x, hit.G.y, hit.G.z,
        hit.C.x, hit.C.y,
        hit.T.x, hit.T.y, hit.T.z,
        hit.B.x, hit.B.y, hit.B.z
    );
    float _output[3];
    dynamic_forward(
        object,
        gathering_map,
        _input,
        _output);
    A = vec3(_output[0], _output[1], _output[2]);
}

void surface_gathering_bw(map_object, GPUPtr gathering_map, vec3 w, Surfel hit, vec3 out_A, vec3 dL_dA)
{
    float _input[20] = float[20](
        w.x, w.y, w.z,
        hit.P.x, hit.P.y, hit.P.z,
        hit.N.x, hit.N.y, hit.N.z,
        hit.G.x, hit.G.y, hit.G.z,
        hit.C.x, hit.C.y,
        hit.T.x, hit.T.y, hit.T.z,
        hit.B.x, hit.B.y, hit.B.z
    );
    float _output[3] = float[3](out_A.x, out_A.y, out_A.z);
    float _output_grad[3] = float[3](dL_dA.x, dL_dA.y, dL_dA.z);
    float _input_grad[20];
    dynamic_backward(
        object,
        gathering_map,
        _input,
        _output,
        _output_grad,
        _input_grad
    );
}
