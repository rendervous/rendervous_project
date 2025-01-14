FORWARD {
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    mat4x3 T = mat4x3_ptr(parameters.transform.data).data[0];
    mat4x3 invT = inverse_transform(T);
    vec3 xt = transform_position(x, invT);
    vec3 wt = transform_direction(w, invT);
    forward(parameters.base_geometry, float[6](xt.x, xt.y, xt.z, wt.x, wt.y, wt.z), _output);
    Surfel s;
    float t;
    int patch_index;
    hit2surfel(x, w, _output, t, patch_index, s);
    s = transform(s, T);
    surfel2array(t, patch_index, s, _output);
}