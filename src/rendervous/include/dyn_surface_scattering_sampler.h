/*
Represents objects that can scatter in surfaces.
*/

void surface_scattering_sampler(map_object, GPUPtr scattering_map, vec3 w, Surfel hit, out vec3 wo, out vec3 W, out float pdf)
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
    float _output[7];
    dynamic_forward(
        object,
        scattering_map,
        _input,
        _output);
    wo = vec3(_output[0], _output[1], _output[2]);
    W = vec3(_output[3], _output[4], _output[5]);
    pdf = _output[6];
}
