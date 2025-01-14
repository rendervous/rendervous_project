float transmittance_DT(map_object, vec3 x, vec3 w)
{
    float tMin, tMax;
    if (!boundary(object, x, w, tMin, tMax))
    return 1.0;

    x += w * tMin;
    float d = tMax - tMin;

    while(true){
        float md; // majorant distance
        float maj = max(0.0000001, majorant(object, x, w, md));

        float dt = -log(1 - random()) / maj;
        float t = min(md, dt);
        x += w * t;
        d -= t;

        if (dt > md)
        continue;

        if (d <= 0.0000001)
            return 1.0;

        if (random() < sigma(object, x)/maj)
            return 0.0;
    }
}

void transmittance_DT_bw(map_object, vec3 x, vec3 w, float dL_dT)
{
    if (dL_dT == 0)
    return;

    float tMin, tMax;
    if (!boundary(object, x, w, tMin, tMax))
    return;

    x += w * tMin;
    float d = tMax - tMin;

    vec3 ini_x = x;
    float ini_d = d;
    uvec4 seed_before_T = get_seed();
    //SAVE_SEED(before_T);
    while(true){
        float md; // majorant distance
        float maj = max(0.0000001, majorant(object, x, w, md) + 0.1);

        float dt = -log(1 - random()) / maj;
        float t = min(md, dt);
        x += w * t;
        d -= t;

        if (dt > md)
        continue;

        if (d <= 0.0000001)
            break;

        if (random() < sigma(object, x)/maj)
            return;
    }

    x = ini_x;
    d = ini_d;
    set_seed(seed_before_T);
    //RESTORE_SEED(before_T);
    while(true){
        float md; // majorant distance
        float maj = max(0.0000001, majorant(object, x, w, md) + 0.1);

        float dt = -log(1 - random()) / maj;
        float t = min(md, dt);
        x += w * t;
        d -= t;

        if (dt > md)
        continue;

        if (d <= 0.0000001)
            break;

        random(); // Only to replay same sequence
        float Pc = sigma(object, x) / maj;

        float dL_dsigma = -dL_dT / maj / (1 - Pc);// / exp(-maj * d);
        sigma_bw(object, x, dL_dsigma);
    }
}
