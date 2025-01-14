#ifndef MATHS_H
#define MATHS_H

#define pi 3.1415926535897932384626433832795
#define piOverTwo 1.5707963267948966192313216916398
#define inverseOfPi 0.31830988618379067153776752674503
#define inverseOfTwoPi 0.15915494309189533576888376337251
#define two_pi 6.283185307179586476925286766559


bool intersect_ray_box(vec3 x, vec3 w, vec3 b_min, vec3 b_max, out float tMin, out float tMax)
{
    // un-parallelize w
    w.x = abs(w).x <= 0.000001 ? 0.000001 : w.x;
    w.y = abs(w).y <= 0.000001 ? 0.000001 : w.y;
    w.z = abs(w).z <= 0.000001 ? 0.000001 : w.z;
    vec3 C_Min = (b_min - x)/w;
    vec3 C_Max = (b_max - x)/w;
	tMin = max(max(min(C_Min[0], C_Max[0]), min(C_Min[1], C_Max[1])), min(C_Min[2], C_Max[2]));
	tMin = max(0.0, tMin);
	tMax = min(min(max(C_Min[0], C_Max[0]), max(C_Min[1], C_Max[1])), max(C_Min[2], C_Max[2]));
	if (tMax <= tMin || tMax <= 0) {
		return false;
	}
	return true;
}

void ray_box_intersection(vec3 x, vec3 w, vec3 b_min, vec3 b_max, out float tMin, out float tMax)
{
    // un-parallelize w
    w.x = abs(w).x <= 0.000001 ? 0.000001 : w.x;
    w.y = abs(w).y <= 0.000001 ? 0.000001 : w.y;
    w.z = abs(w).z <= 0.000001 ? 0.000001 : w.z;
    vec3 C_Min = (b_min - x)/w;
    vec3 C_Max = (b_max - x)/w;
	tMin = max(max(min(C_Min[0], C_Max[0]), min(C_Min[1], C_Max[1])), min(C_Min[2], C_Max[2]));
	tMax = min(min(max(C_Min[0], C_Max[0]), max(C_Min[1], C_Max[1])), max(C_Min[2], C_Max[2]));
}


// https://github.com/google/spherical-harmonics


#define _C0 0.28209479177387814
#define _C1 0.4886025119029199
#define _C20 1.0925484305920792
#define _C21 -1.0925484305920792
#define _C22 0.31539156525252005
#define _C23 -1.0925484305920792
#define _C24 0.5462742152960396

//
//const float C3[7] = float[7](
//    -0.5900435899266435,
//    2.890611442640554,
//    -0.4570457994644658,
//    0.3731763325901154,
//    -0.4570457994644658,
//    1.445305721320277,
//    -0.5900435899266435
//);
//
//const float C4[9] = float[9](
//     2.5033429417967046,
//     -1.7701307697799304,
//     0.9461746957575601,
//     -0.6690465435572892,
//     0.10578554691520431,
//     -0.6690465435572892,
//     0.47308734787878004,
//     -1.7701307697799304,
//     0.6258357354491761
// );

void eval_sh(vec3 w, out float coef[1])
{
    coef[0] = _C0;
}

void eval_sh(vec3 w, out float coef[4])
{
    coef[0] = _C0;
    float x = w.x, y = w.y, z = w.z;
    float xx = x * x, yy = y * y, zz = z * z;
    float xy = x * y, yz = y * z, xz = x * z;
    coef[1] = -_C1 * y;
    coef[2] = _C1 * z;
    coef[3] = -_C1 * x;
}

void eval_sh(vec3 w, out float coef[9])
{
    coef[0] = _C0;
    float x = w.x, y = w.y, z = w.z;
    float xx = x * x, yy = y * y, zz = z * z;
    float xy = x * y, yz = y * z, xz = x * z;
    coef[4] = _C20 * xy;
    coef[5] = _C21 * yz;
    coef[6] = _C22 * (2.0 * zz - xx - yy);
    coef[7] = _C23 * xz;
    coef[8] = _C24 * (xx - yy);
    coef[1] = -_C1 * y;
    coef[2] = _C1 * z;
    coef[3] = -_C1 * x;
}

//void eval_sh16(vec3 w, out float coef[16])
//{
//    coef[0] = C0;
//    float x = w.x, y = w.y, z = w.z;
//    float xx = x * x, yy = y * y, zz = z * z;
//    float xy = x * y, yz = y * z, xz = x * z;
//    coef[4] = C2[0] * xy;
//    coef[5] = C2[1] * yz;
//    coef[6] = C2[2] * (2.0 * zz - xx - yy);
//    coef[7] = C2[3] * xz;
//    coef[8] = C2[4] * (xx - yy);
//    coef[1] = -C1 * y;
//    coef[2] = C1 * z;
//    coef[3] = -C1 * x;
//    coef[9] = C3[0] * y * (3 * xx - yy);
//    coef[10] = C3[1] * xy * z;
//    coef[11] = C3[2] * y * (4 * zz - xx - yy);
//    coef[12] = C3[3] * z * (2 * zz - 3 * xx - 3 * yy);
//    coef[13] = C3[4] * x * (4 * zz - xx - yy);
//    coef[14] = C3[5] * z * (xx - yy);
//    coef[15] = C3[6] * x * (xx - 3 * yy);
//}


mat4 LookAtLH(vec3 camera, vec3 target, vec3 upVector)
{
	vec3 zaxis = normalize(target - camera);
	vec3 xaxis = normalize(cross(upVector, zaxis));
	vec3 yaxis = cross(zaxis, xaxis);

	return mat4(
		xaxis.x, yaxis.x, zaxis.x, 0,
		xaxis.y, yaxis.y, zaxis.y, 0,
		xaxis.z, yaxis.z, zaxis.z, 0,
		-dot(xaxis, camera), -dot(yaxis, camera), -dot(zaxis, camera), 1);
}

mat4 PerspectiveFovLH(float fieldOfView, float aspectRatio, float znearPlane, float zfarPlane)
{
	float h = 1.0 / tan(fieldOfView / 2.0);
	float w = h * aspectRatio;

	return mat4(
		w, 0, 0, 0,
		0, h, 0, 0,
		0, 0, zfarPlane / (zfarPlane - znearPlane), 1,
		0, 0, -znearPlane * zfarPlane / (zfarPlane - znearPlane), 0);
}

vec3 usc2dir(vec2 angles)
{
    float y = cos(angles.y);
    float s = sin(angles.y);
    float x = sin(angles.x) * s;
    float z = cos(angles.x) * s;
    return vec3(x, y, z);
}

vec2 dir2usc(vec3 w) {
    w.x += 0.0000001 * int(w.x == 0.0 && w.z == 0.0);
    float beta = acos(clamp(w.y, -1.0, 1.0));
    float alpha = atan(w.x, w.z);
    return vec2(alpha, beta);
}

vec2 dir2xr(vec3 w)
{
    w.x += 0.0000001 * int(w.x == 0.0 && w.z == 0.0);
    float x = atan(w.x, w.z) * inverseOfPi;
    float y = -2 * asin(clamp(w.y, -1.0, 1.0)) * inverseOfPi;
    return vec2(x, y);
}

vec3 xr2dir(vec2 c)
{
    vec2 angles = vec2(c.x * pi, c.y * piOverTwo);
    float y = -sin(angles.y);
    float r = cos(angles.y);
    float x = sin(angles.x) * r;
    float z = cos(angles.x) * r;
    return vec3(x, y, z);
}


// Adapted from:
// https://gamedev.stackexchange.com/questions/169508/octahedral-impostors-octahedral-mapping

vec2 dir2oct(vec3 w)
{
    vec3 octant = sign(w);
    // Scale the vector so |x| + |y| + |z| = 1 (surface of octahedron).
    float sum = dot(w, octant);
    vec3 octahedron = w / sum;

    // "Untuck" the corners using the same reflection across the diagonal as before.
    // (A reflection is its own inverse transformation).
    if(octahedron.z < 0) {
        vec3 absolute = abs(octahedron);
        octahedron.xy = octant.xy
                      * vec2(1.0 - absolute.y, 1.0 - absolute.x);
    }
    return octahedron.xy;
}

vec3 oct2dir(vec2 c)
{
    // Unpack the 0...1 range to the -1...1 unit square.
    vec3 w = vec3(c, 0);

    // "Lift" the middle of the square to +1 z, and let it fall off linearly
    // to z = 0 along the Manhattan metric diamond (absolute.x + absolute.y == 1),
    // and to z = -1 at the corners where position.x and .y are both = +-1.
    vec2 absolute = abs(w.xy);
    w.z = 1.0 - absolute.x - absolute.y;

    // "Tuck in" the corners by reflecting the xy position along the line y = 1 - x
    // (in quadrant 1), and its mirrored image in the other quadrants.
    if(w.z < 0)
        w.xy = sign(w.xy)
                    * vec2(1.0 - absolute.y, 1.0 - absolute.x);

    return normalize(w);
}


// MORTON CODE FUNCTIONS
// Adapted from https://gist.github.com/wontonst/8696dcfb643121c864dec7c0d6ad26c5

int part1by1(int n){
    n &= 0x0000ffff;
    n = (n | (n << 8)) & 0x00FF00FF;
    n = (n | (n << 4)) & 0x0F0F0F0F;
    n = (n | (n << 2)) & 0x33333333;
    n = (n | (n << 1)) & 0x55555555;
    return n;
}

int unpart1by1(int n){
    n &= 0x55555555; // base10: 1431655765, binary: 1010101010101010101010101010101,  len: 31
    n = (n ^ (n >> 1)) & 0x33333333; // base10: 858993459,  binary: 110011001100110011001100110011,   len: 30
    n = (n ^ (n >> 2)) & 0x0f0f0f0f; // base10: 252645135,  binary: 1111000011110000111100001111,     len: 28
    n = (n ^ (n >> 4)) & 0x00ff00ff; // base10: 16711935,   binary: 111111110000000011111111,         len: 24
    n = (n ^ (n >> 8)) & 0x0000ffff; // base10: 65535,      binary: 1111111111111111,                 len: 16
    return n;
}

void morton2pixel(int index, out int px, out int py)
{
    px = unpart1by1(index);
    py = unpart1by1(index >> 1);
}

ivec2 morton2pixel(int index)
{
    return ivec2(unpart1by1(index), unpart1by1(index >> 1));
}

void pixel2morton(int px, int py, out int index)
{
    index = part1by1(px) | (part1by1(py) << 1);
}

int pixel2morton(ivec2 px)
{
    return part1by1(px.x) | (part1by1(px.y) << 1);
}


#endif