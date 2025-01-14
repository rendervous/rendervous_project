
void environment(map_object, vec3 x, vec3 w, out vec3 E);
void environment_bw(map_object, vec3 x, vec3 w, vec3 dL_dE);


FORWARD {{
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    vec3 E;
    environment(object, x, w, E);
    _output = float[3](E.x, E.y, E.z);
}}

BACKWARD {{
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    vec3 dL_dE = vec3(_output_grad[0], _output_grad[1], _output_grad[2]);
    environment_bw(object, x, w, dL_dE);
}}