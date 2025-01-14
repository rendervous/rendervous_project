#include "sc_visibility.h"
#include "sc_environment.h"

FORWARD {
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    BEGIN_BRANCH(E)
    vec3 E;
    environment(object, x, w, E);
    END_BRANCH(E)
    BEGIN_BRANCH(T)
    float T = ray_visibility(object, x, w);
    END_BRANCH(T)

    vec3 Tr = T * E;
    _output = float[3](Tr.x, Tr.y, Tr.z);
}

BACKWARD_USING_OUTPUT {
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    vec3 Tr = vec3(_output[0], _output[1], _output[2]);
    vec3 dL_dTr = vec3(_output_grad[0], _output_grad[1], _output_grad[2]);
    BEGIN_BRANCH(E)
    vec3 E;
    SAVE_SEED(before_E)
    environment(object, x, w, E);
    vec3 Tc = Tr / E;
    Tc = mix(Tc, vec3(0.0), isnan(Tc));
    float T = max(Tc.x, max(Tc.y, Tc.z));
    SET_SEED(before_E)
    environment_bw(object, x, w, dL_dTr * T);
    END_BRANCH(E)
    float dL_dT = dot(dL_dTr, E);
    BEGIN_BRANCH(T)
    ray_visibility_bw(object, x, w, T, dL_dT);
    END_BRANCH(T)

    // TODO: NO INPUT GRAD!
}
