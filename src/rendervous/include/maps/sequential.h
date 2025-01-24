/*
Assumes generics to be:
INPUT_DIM
INTERMEDIATE_DIM
OUTPUT_DIM
NUMBER_OF_MAPS

Assumes the parameters to be:
GPUPtr initial_map; // from INPUT_DIM -> INTERMEDIATE_DIM
GPUPtr intermediate_maps[NUMBER_OF_MAPS];  // from INTERMEDIATE_DIM -> INTERMEDIATE_DIM
GPUPtr final_map;  // from INTERMEDIATE_DIM -> OUTPUT_DIM
*/

//FORWARD
//{
//    float intermediate_0[MAP0_OUTPUT_DIM];
//    forward(parameters.map_0, _input, intermediate_0);
//    float intermediate_1[MAP1_OUTPUT_DIM];
//    forward(parameters.map_1, intermediate_0, intermediate_1);
//    //.
//    forward(parameters.map_2, intermediate_1, _output);
//}

FORWARD
{
    float intermediate_values[2][INTERMEDIATE_DIM];
    forward(parameters.initial_map, _input, intermediate_values[0]);
    [[unroll]] for (int layer = 0; layer < NUMBER_OF_MAPS; layer ++)
        dynamic_forward(object, parameters.maps[layer], intermediate_values[layer % 2], intermediate_values[(layer + 1) % 2]);
    forward(parameters.final_map, intermediate_values[NUMBER_OF_MAPS % 2], _output);
}
