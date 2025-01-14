
void volume_gathering(map_object, vec3 x, vec3 w, out vec3 A);
void volume_gathering_bw(map_object, vec3 x, vec3 w, vec3 out_A, vec3 dL_dA);

FORWARD {
   vec3 x = vec3(_input[0], _input[1], _input[2]);
   vec3 w = vec3(_input[3], _input[4], _input[5]);
   vec3 A;
   BEGIN_BRANCH(G)
   volume_gathering(object, x, w, A);
   END_BRANCH(G)
   _output = float[3](A.x, A.y, A.z);
}

BACKWARD_USING_OUTPUT {
   vec3 x = vec3(_input[0], _input[1], _input[2]);
   vec3 w = vec3(_input[3], _input[4], _input[5]);
   vec3 A = vec3(_output[0], _output[1], _output[2]);
   vec3 dL_dA = vec3(_output_grad[0], _output_grad[1], _output_grad[2]);
   BEGIN_BRANCH(G)
   volume_gathering_bw(object, x, w, A, dL_dA);
   END_BRANCH(G)
   // TODO: Grad for INPUTS ?
}
