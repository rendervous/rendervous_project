#include "sc_surfaces.h"
#include "sc_boundaries.h"
#include "dyn_surface_scattering_sampler.h"
#include "dyn_medium_traversal.h"
#include "sc_path_sampler.h"
//#include "sc_path_tracing_SPS.h"


FORWARD {
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 win = vec3(_input[3], _input[4], _input[5]);
    vec3 w, W;
    path_sampler(object, x, win, w, W);
    _output = float[6](w.x, w.y, w.z, W.x, W.y, W.z);
}

