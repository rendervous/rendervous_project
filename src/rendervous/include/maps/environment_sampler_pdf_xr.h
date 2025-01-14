#include "environment_sampler_pdf_interface.h"

void environment_sampler_pdf(map_object, vec3 x, vec3 w, out float pdf)
{
    // compute w converted to pixel px, py and to z-order index
    vec2 xr = dir2xr(w);
    vec2 c = xr * 0.5 + vec2(0.5); // 0,0 - 1,1
    ivec2 p = ivec2(clamp(c, vec2(0.0), vec2(0.999999)) * (1 << parameters.levels));
    int index = pixel2morton(p);
    // compute last level offset 4*(4^(levels - 1) - 1)/3 and peek pdf of the cell
    pdf = 1.0;
    if (parameters.levels > 0) // more than one node
    {
        int offset = 4 * (1 << (2 * (parameters.levels - 1)) - 1) / 3;
        // peek the density and compute final point pdf
        float_ptr densities_buf = float_ptr(parameters.densities);
        pdf = densities_buf.data[offset + index];
    }
    // multiply by peek in that area
    vec2 p0 = vec2(p) / (1 << parameters.levels);
    vec2 p1 = vec2(p + ivec2(1)) / (1 << parameters.levels);
    float pixel_area = 2 * pi * (cos(p0.y*pi) - cos(p1.y*pi)) / (1 << parameters.levels);
    pdf *= 1.0 / pixel_area;
}