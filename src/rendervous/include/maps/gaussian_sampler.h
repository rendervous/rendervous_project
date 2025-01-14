FORWARD
{
    float sum = 0;
    [[unroll]] for (int i=0; i<(OUTPUT_DIM - 1) / 2; i++)
    {
        vec2 r = gauss2();
        _output[i*2] = r.x;
        _output[i*2 + 1] = r.y;
        sum += dot(r, r);
    }
    #if (OUTPUT_DIM % 2 == 0)
    {
        float r = gauss();
        _output[OUTPUT_DIM-2] = r;
        sum += r * r;
    }
    #endif
    _output[OUTPUT_DIM-1] = pow(two_pi, 0.5 * (OUTPUT_DIM-1)) * exp (0.5 * sum); // 1 / pdf(x)
}