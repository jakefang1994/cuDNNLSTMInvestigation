#include <THC.h>
#include <THCGeneral.h>

#include "lstm_kernel.h"

extern THCState* state;

float lstm_forward(
    THCudaTensor* h_data,
    THCudaTensor* x_data,
    THCudaTensor* c_data,
    int hiddenSize, int miniBatch, int seqLength, int numLayers) {

    return forward(state, h_data, x_data, c_data, 
                hiddenSize, miniBatch, seqLength, numLayers);
}
