FORWARD
{
    [[unroll]] for (int i=0; i<OUTPUT_DIM - 1; i++) _output[i] = random();
    _output[OUTPUT_DIM-1] = 1.0;
}