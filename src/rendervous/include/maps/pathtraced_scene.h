#include "sc_environment.h"
#include "sc_surfaces.h"
#include "sc_boundaries.h"
#include "dyn_surface_scattering_sampler.h"
#include "dyn_surface_gathering.h"
#include "dyn_medium_traversal.h"
#include "sc_path_tracing.h"
//#include "sc_path_tracing_SPS.h"


FORWARD {
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    vec3 A;
    path_trace(object, x, w, A);
    _output = float[3](A.x, A.y, A.z);
}

BACKWARD {
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    vec3 dL_dA = vec3(_output_grad[0], _output_grad[1], _output_grad[2]);
    vec3 A;

    SAVE_SEED(before_path);
    path_trace_fw(object, x, w, A);
    SAVE_SEED(after_path);

    SET_SEED(before_path);
    path_trace_bw(object, x, w, A, dL_dA);
    ASSERT(get_seed() == SEED(after_path), "pathtraced_scene.h: Seed inconsistency.")
}