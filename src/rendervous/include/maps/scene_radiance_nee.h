#include "vr_environment.h"
#include "sc_surfaces.h"
#include "sc_surface_scattering.h"
#include "sc_surface_scattering_sampler_pdf.h"
#include "sc_surface_scattering_sampler.h"
#include "sc_emitters.h"
#include "sc_emitters_sampler_pdf.h"
#include "sc_emitters_sampler.h"
#include "sc_surface_emission.h"
#include "sc_volume_scattering.h"
#include "sc_path_integrator_PT_NEE.h"


FORWARD {
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    vec3 R = path_trace_NEE(object, x, w);
    _output = float[3](R.x, R.y, R.z);
}

BACKWARD {
}
