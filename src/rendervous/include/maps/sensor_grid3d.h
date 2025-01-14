FORWARD
{
    ivec3 index = floatBitsToInt(vec3(_input[0], _input[1], _input[2]));
    vec3 subsample = vec3(0.0);
    if (parameters.sd != 0)
        subsample = gauss3() * parameters.sd;
    vec3 sx = vec3((index[2] + subsample.x) / (parameters.width - 1), (index[1] + subsample.y) / (parameters.height - 1), (index[0] + subsample.z) / (parameters.depth - 1));
    sx = (parameters.box_max - parameters.box_min) * sx + parameters.box_min;
    _output = float[3]( sx.x, sx.y, sx.z );
}
