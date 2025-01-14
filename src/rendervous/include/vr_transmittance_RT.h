float transmittance_RT(map_object, vec3 x, vec3 w)
{
    float T = 1.0;

    float tMin, tMax;
    if (!boundary(object, x, w, tMin, tMax))
    return T;

    x += w * tMin;
    float d = tMax - tMin;

    while(true){
        float md; // majorant distance
        float maj = max(0.0000001, majorant(object, x, w, md));

        float dt = min(-log(1 - random()) / maj, min(d, md));
        x += w * dt;
        d -= dt;
        if (d <= 0.0000001)
            return T;

        T *= (1 - sigma(object, x)/maj);

        if (T < 0.01)
        {
            if (random() < 1 - T)
                return 0;
            T = 1.0; // T / T
        }

//        if (T < 0.001)
//        {
//            if (random() < 1 - T)
//                return 0;
//            T = 1.0; // T / T
//        }

//        if (T < 0.1)
//            if (random() < 1 - T)
//                return 0;
//            else
//                T = 1.0;
//        if (T < 0.01)
//            if (random() < 0.99)
//                return 0;
//            else
//                T *= 100;
    }
}

void transmittance_RT_bw(map_object, vec3 x, vec3 w, float dL_dT)
{
    if (dL_dT == 0)
    return;

    float tMin, tMax;
    if (!boundary(object, x, w, tMin, tMax))
    return;

    float EPS = 0.0000001;

    float T = 1.0;

    x += w * tMin;
    float d = tMax - tMin;

    vec3 ini_x = x;
    float ini_d = d;

//    SAVE_SEED(before_T);
    uvec4 seed_before_T = get_seed();

    while(true){
        float md; // majorant distance
        float maj = max(EPS, majorant(object, x, w, md) + 0.1);

        float dt = min(-log(1 - random()) / maj, min(d, md));
        x += w * dt;
        d -= dt;
        if (d <= EPS)
            break;

        T *= (1 - sigma(object, x)/maj);

        if (T < 0.1)
        {
            if (random() < 1 - T)
                return;
            T = 1.0; // T / T
        }

//        if (T < 0.01)
//            if (random() < 0.99)
//                return;
//            else
//                T *= 100;
    }

    float out_T = T;

    T = 1.0;

//    RESTORE_SEED(before_T);
    set_seed(seed_before_T);
    x = ini_x;
    d = ini_d;
    while(true){
        float md; // majorant distance
        float maj = max(EPS, majorant(object, x, w, md) + 0.1);

        float dt = min(-log(1 - random()) / maj, min(d, md));
        x += w * dt;
        d -= dt;
        if (d <= EPS)
            break;

        float r = (1 - sigma(object, x)/maj);
        T *= r;

        float dL_dsigma = - out_T * dL_dT / maj / r;
        sigma_bw(object, x, dL_dsigma);

        if (T < 0.1)
        {
            advance_random(); // will never hit eval true body...
//            if (random() < 1 - T)
//                return;
            T = 1.0; // T / T
        }

//        if (T < 0.01)
//        {
//            advance_random();
//            T *= 100;
////            if (random() < 0.99)
////                return;
////            else
////                T *= 100;
//        }
    }
}
