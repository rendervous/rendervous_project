#ifndef SCATTERING_AD_H
#define SCATTERING_AD_H

vec3 hg_phase_sample(vec3 w_in, float g) {
	float phi = random() * 2 * pi;
    float xi = random();
    float g2 = g * g;
    float one_minus_g2 = 1.0 - g2;
    float one_plus_g2 = 1.0 + g2;
    float one_over_2g = 0.5 / g;

	float t = one_minus_g2 / (1.0f - g + 2.0f * g * xi);
	float invertcdf = one_over_2g * (one_plus_g2 - t * t);
	float cosTheta = abs(g) < 0.001 ? 2 * xi - 1 : invertcdf;
	float sinTheta = sqrt(max(0, 1.0f - cosTheta * cosTheta));
	vec3 t0, t1;
	createOrthoBasis(w_in, t0, t1);
    return sinTheta * sin(phi) * t0 + sinTheta * cos(phi) * t1 + cosTheta * w_in;
}

vec3 hg_phase_sample(vec3 w_in, float g, out float pdf) {
	float phi = random() * 2 * pi;
    float xi = random();
    float g2 = g * g;
    float one_minus_g2 = 1.0 - g2;
    float one_plus_g2 = 1.0 + g2;
    float one_over_2g = 0.5 / g;

	float t = one_minus_g2 / (1.0f - g + 2.0f * g * xi);
	float invertcdf = one_over_2g * (one_plus_g2 - t * t);
	float cosTheta = abs(g) < 0.001 ? 2 * xi - 1 : invertcdf;
	float sinTheta = sqrt(max(0, 1.0f - cosTheta * cosTheta));
	pdf = 0.25 / pi * (one_minus_g2) / pow(one_plus_g2 - 2 * g * cosTheta, 1.5);
	vec3 t0, t1;
	createOrthoBasis(w_in, t0, t1);
    return sinTheta * sin(phi) * t0 + sinTheta * cos(phi) * t1 + cosTheta * w_in;
}

float hg_phase_eval(float cos_theta, float g)
{
	if (abs(g) < 0.001)
		return 0.25 / pi;
    float g2 = g * g;
    float one_minus_g2 = 1.0 - g2;
    float one_plus_g2 = 1.0 + g2;
	return 0.25 / pi * (one_minus_g2) / pow(one_plus_g2 - 2 * g * cos_theta, 1.5);
}

float hg_phase_eval(vec3 w_in, vec3 w_out, float g)
{
    return hg_phase_eval(dot(w_in, w_out), g);
}


#endif