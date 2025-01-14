float transmittance_RT(map_object, vec3 x, vec3 w, float d)
{
    float T = 1.0;
    float maj = (parameters.majorant + 0.1)*1.05;
    while(true){
        float dt = min(-log(1 - random()) / maj, min(d, md));
        x += w * dt;
        d -= dt;
        if (d <= 0.0000001)
            return T;
        T *= (1 - sigma(object, x)/maj);
    }
    return T;
}

void transmittance_RT_bw(map_object, vec3 x, vec3 w, float d, float out_T, float dL_dT)
{
    float maj = (parameters.majorant + 0.1)*1.05;
    while(true){
        float dt = min(-log(1 - random()) / maj, min(d, md));
        x += w * dt;
        d -= dt;
        if (d <= 0.0000001)
            return T;
        float r = (1 - sigma(object, x)/maj);
        float dL_dsigma = - out_T * dL_dT / maj / r;
        sigma_bw(object, x, dL_dsigma);
    }
}
