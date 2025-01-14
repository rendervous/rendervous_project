FORWARD
{
    ivec3 index = floatBitsToInt(vec3(_input[0], _input[1], _input[2]));
    vec3_ptr poses_buf = vec3_ptr(parameters.poses);
    vec3 o = poses_buf.data[index[0]*3 + 0];
    vec3 d = poses_buf.data[index[0]*3 + 1];
    vec3 n = poses_buf.data[index[0]*3 + 2];

    vec2 subsample = vec2(0.5);
    if (parameters.generation_mode == 1)
        subsample = vec2(random(), random());

    float sx = ((index[2] + subsample.x) * 2 - parameters.width) * parameters.znear / parameters.height;
    float sy = ((index[1] + subsample.y) * 2 - parameters.height) * parameters.znear / parameters.height;
    float sz = parameters.znear / tan(parameters.fov * 0.5);

    vec3 zaxis = normalize(d);
    vec3 xaxis = normalize(cross(n, zaxis));
    vec3 yaxis = cross(zaxis, xaxis);

    vec3 x, w;
    w = xaxis * sx + yaxis * sy + zaxis * sz;
    x = o + w;
    w = normalize(w);

    _output = float[6]( x.x, x.y, x.z, w.x, w.y, w.z );
}
