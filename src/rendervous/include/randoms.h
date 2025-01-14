#ifndef RANDOMS_H
#define RANDOMS_H

#include "maths.h"

uint TausStep(uint z, int S1, int S2, int S3, uint M) { uint b = (((z << S1) ^ z) >> S2); return ((z & M) << S3) ^ b; }

uint LCGStep(uint z, uint A, uint C) { return A * z + C; }

uvec4 rdv_rng_state;


void set_seed(uvec4 seed) {
    rdv_rng_state = seed;
}

uvec4 get_seed() {
    return rdv_rng_state;
}



//float random()
//{
//    float f;
//    do{
//	rdv_rng_state.x = TausStep(rdv_rng_state.x, 13, 19, 12, 4294967294U);
//	rdv_rng_state.y = TausStep(rdv_rng_state.y, 2, 25, 4, 4294967288U);
//	rdv_rng_state.z = TausStep(rdv_rng_state.z, 3, 11, 17, 4294967280U);
//	rdv_rng_state.w = LCGStep(rdv_rng_state.w, 1664525, 1013904223U);
////	if (!rnd_antithetic)
//        //     2.3283064364387
//	    f = 2.3283064364387e-10 * uint(rdv_rng_state.x ^ rdv_rng_state.y ^ rdv_rng_state.z ^ rdv_rng_state.w); // THERE WAS AN ERROR HERE!
//	    //return mod(4.294967296e-10 * (rng_state.x ^ rng_state.y ^ rng_state.z ^ rng_state.w), 1.0);
////	else
////	    return 2.3283064365387e-10 * ~(rng_state.x ^ rng_state.y ^ rng_state.z ^ rng_state.w);
//    } while(f < 0.0 || f >= 1.0);
//    return f;
//}

void advance_random(inout uvec4 rdv_rng_state)
{
	rdv_rng_state.x = TausStep(rdv_rng_state.x, 13, 19, 12, 4294967294U);
	rdv_rng_state.y = TausStep(rdv_rng_state.y, 2, 25, 4, 4294967288U);
	rdv_rng_state.z = TausStep(rdv_rng_state.z, 3, 11, 17, 4294967280U);
	rdv_rng_state.w = LCGStep(rdv_rng_state.w, 1664525, 1013904223U);
}

void advance_random(){
    advance_random(rdv_rng_state);
}

float random()
{
    advance_random();
	uint v = rdv_rng_state.x ^ rdv_rng_state.y ^ rdv_rng_state.z ^ rdv_rng_state.w;
	return uintBitsToFloat(v & 0x007FFFFF | 0x3F800000) - 1;
    // f = 2.3283064364387e-10 * uint(rdv_rng_state.x ^ rdv_rng_state.y ^ rdv_rng_state.z ^ rdv_rng_state.w); // THERE WAS AN ERROR HERE!
}

vec2 rdv_BM() {
	float u1 = 1.0 - random(); //uniform(0,1] random doubles
	float u2 = 1.0 - random();
	float r = sqrt(-2.0 * log(max(0.0000000001, u1)));
	float t = 2.0 * pi * u2;
	return r * vec2(cos(t), sin(t));
}

float gauss() {
	return rdv_BM().x;
}

vec2 gauss2()
{
    return rdv_BM();
}

vec3 gauss3()
{
    return vec3(rdv_BM(), rdv_BM().x);
}

vec4 gauss4()
{
    return vec4(rdv_BM(), rdv_BM());
}

float gauss(float mu, float sd)
{
    return sd * gauss() + mu;
}

vec2 gauss(vec2 mu, vec2 sd)
{
    return sd * gauss2() + mu;
}

vec3 gauss(vec3 mu, vec3 sd)
{
    return sd * gauss3() + mu;
}

vec4 gauss(vec4 mu, vec4 sd)
{
    return sd * gauss4() + mu;
}

/*
Useful to spawn secondary seed generations saving the previous seed
*/
uvec4 set_hash_seed()
{
    uvec4 main_seed = rdv_rng_state;

    // TODO: Some hash, we should do some stats test on this...
    rdv_rng_state = (~rdv_rng_state + 13) ^ (((rdv_rng_state.x + 11)*17 + rdv_rng_state.y*39)+11) * rdv_rng_state.z;
    random(); // decorrelate a little

    return main_seed;
}

uvec4 create_branch_seed()
{
    uvec4 s = floatBitsToUint(vec4(random(), random(), random(), random())) + 129;
    advance_random(s);
    advance_random(s);
    advance_random(s);
    advance_random(s);
    return s;
}

uvec4 begin_branch_seed()
{
    uvec4 s = floatBitsToUint(vec4(random(), random(), random(), random())) + 129;
    uvec4 main_seed = rdv_rng_state;
    rdv_rng_state = s;
    advance_random();
    advance_random();
    advance_random();
    advance_random();
    return main_seed;
}

void createOrthoBasis(vec3 N, out vec3 T, out vec3 B)
{
    float sign = N.z >= 0 ? 1 : -1;
    float a = -1.0f / (sign + N.z);
    float b = N.x * N.y * a;
    T = vec3(1.0f + sign * N.x * N.x * a, sign * b, -sign * N.x);
    B = vec3(b, sign + N.y * N.y * a, -N.y);
}

vec3 randomDirection(vec3 D) {
	float r1 = random();
	float r2 = random() * 2 - 1;
	float sqrR2 = r2 * r2;
	float two_pi_by_r1 = two_pi * r1;
	float sqrt_of_one_minus_sqrR2 = sqrt(max(0, 1.0 - sqrR2));
	float x = cos(two_pi_by_r1) * sqrt_of_one_minus_sqrR2;
	float y = sin(two_pi_by_r1) * sqrt_of_one_minus_sqrR2;
	float z = r2;
	vec3 t0, t1;
	createOrthoBasis(D, t0, t1);
	return t0 * x + t1 * y + D * z;
}

vec3 randomDirection()
{
    return randomDirection(vec3(0.0, 1.0, 0.0));
}

vec3 randomDirection(vec3 D, float fov) {
	float r1 = random();
	float r2 = 1 - random() * (1 - cos(fov));
	float sqrR2 = r2 * r2;
	float two_pi_by_r1 = two_pi * r1;
	float sqrt_of_one_minus_sqrR2 = sqrt(max(0, 1.0 - sqrR2));
	float x = cos(two_pi_by_r1) * sqrt_of_one_minus_sqrR2;
	float y = sin(two_pi_by_r1) * sqrt_of_one_minus_sqrR2;
	float z = r2;
	vec3 t0, t1;
	createOrthoBasis(D, t0, t1);
	return t0 * x + t1 * y + D * z;
}

vec3 randomDirection(float alpha0, float alpha1, float beta0, float beta1) {
	float r1 = random() * (alpha1 - alpha0) + alpha0;
	float r2 = cos(beta0) - random() * (cos(beta0) - cos(beta1));
	float sqrR2 = r2 * r2;
	float two_pi_by_r1 = r1;
	float sqrt_of_one_minus_sqrR2 = sqrt(max(0, 1.0 - sqrR2));
	float x = sin(two_pi_by_r1) * sqrt_of_one_minus_sqrR2;
	float y = cos(two_pi_by_r1) * sqrt_of_one_minus_sqrR2;
	float z = r2;
	vec3 t0 = vec3(1, 0, 0);
	vec3 t1 = vec3(0, 0, 1);
	vec3 D = vec3(0, 1, 0);
	return t0 * x + t1 * y + D * z;
}

vec3 randomHSDirection(vec3 D) {
	float r1 = random();
	float r2 = random();
	float sqrR2 = r2 * r2;
	float two_pi_by_r1 = two_pi * r1;
	float sqrt_of_one_minus_sqrR2 = sqrt(max(0, 1.0 - sqrR2));
	float x = cos(two_pi_by_r1) * sqrt_of_one_minus_sqrR2;
	float y = sin(two_pi_by_r1) * sqrt_of_one_minus_sqrR2;
	float z = r2;
	vec3 t0, t1;
	createOrthoBasis(D, t0, t1);
	return t0 * x + t1 * y + D * z;
}

vec3 randomHSDirectionCosineWeighted(vec3 N, out float NdotD)
{
	vec3 t0, t1;
	createOrthoBasis(N, t0, t1);

	while (true) {
		float x = random() * 2 - 1;
		float y = random() * 2 - 1;
		float d2 = x * x + y * y;
		if (d2 > 0.001 && d2 < 1)
		{
			float z = sqrt(1 - d2);
			NdotD = z;
			return t0 * x + t1 * y + N * z;
		}
	}
	return vec3(0,0,0);
}

#endif