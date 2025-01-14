/*
Represents objects that can scatter in surfaces.
*/

void surface_scattering(map_object, GPUPtr scattering_map, vec3 wi, vec3 wo, Surfel hit, out vec3 W)
{
    float _input[23] = float[23](
        wi.x, wi.y, wi.z,
        wo.x, wo.y, wo.z,
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
        scattering_map,
        _input,
        _output);
    W = vec3(_output[0], _output[1], _output[2]);
}
