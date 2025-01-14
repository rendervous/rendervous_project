#ifndef COMMON_FUNCTIONS_H
#define COMMON_FUNCTIONS_H

#include "../common.h"

#ifndef LOCAL_SIZE_X
#define LOCAL_SIZE_X 1024
#endif

#ifndef LOCAL_SIZE_Y
#define LOCAL_SIZE_Y 1
#endif

#ifndef LOCAL_SIZE_Z
#define LOCAL_SIZE_Z 1
#endif

layout (local_size_x = LOCAL_SIZE_X, local_size_y = LOCAL_SIZE_Y, local_size_z = LOCAL_SIZE_Z) in;

layout(binding = 0) uniform system_info {
    uvec4 rdv_seeds;
    int dim_x;
    int dim_y;
    int dim_z;
};

bool start_function(out uvec3 thread_id)
{
    thread_id = gl_GlobalInvocationID;

    if (any(greaterThanEqual(thread_id, uvec3(dim_x, dim_y, dim_z))))
        return false;

    int index = int(thread_id.x + (thread_id.z * dim_y + thread_id.y) * dim_x);

    uvec4 current_seeds = rdv_seeds + uvec4(0x23F1,0x3137,129,index + 129) ;//^ uvec4(int(cos(index)*1000000), index ^ 1231231, index + 1234122, index + seeds.w * 100202021);//seeds ^ uvec4(index ^ 17, index * 123111171, index + 11, index ^ (seeds.x + 13 * seeds.y));

    //uvec4 current_seeds = seeds ^ uvec4(index * 78182311, index ^ 1231231, index + 1234122, index + seeds.w * 100202021);//seeds ^ uvec4(index ^ 17, index * 123111171, index + 11, index ^ (seeds.x + 13 * seeds.y));
    set_seed(current_seeds);
    random();
    random();
    random();
    random();
    uvec4 new_seed = floatBitsToUint(vec4(random(), random(), random(), random()));
    set_seed(new_seed);

//    uvec4 current_seeds = rdv_seeds ^ uvec4(index ^ 17, index * 123111171, index + 11, index ^ (rdv_seeds.x + 13 * rdv_seeds.y));
//    set_seed(current_seeds);
//    random();
//    random();
//    random();
    return true;
}

#endif