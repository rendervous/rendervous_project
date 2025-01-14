#include "volume_gathering_interface.h"
#include "vr_emission.h"
#include "vr_scattering_albedo.h"

void volume_gathering(map_object, vec3 x, vec3 w, out vec3 A)
{
    vec3 E = emission(object, x, w, A);
    vec3 S = scattering_albedo(object, x);
    A = (1 - S)*E;
}

void volume_gathering_bw(map_object, vec3 x, vec3 w, vec3 out_A, vec3 dL_dA)
{
    vec3 E = emission(object, x, w, A);
    vec3 S = scattering_albedo(object, x);
    // A = (1 - S)*E;
    emission_bw(object, x, w, dL_dA*(1 - S));
    scattering_albedo_bw(object, x, dL_dA*E);
}
