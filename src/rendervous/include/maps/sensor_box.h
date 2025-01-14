FORWARD
{
    vec3 sx = (parameters.box_max - parameters.box_min) * vec3(random(), random(), random()) + parameters.box_min;
    _output = float[3]( sx.x, sx.y, sx.z );
}