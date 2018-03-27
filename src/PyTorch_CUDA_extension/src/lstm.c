#include <THC.h>
#include <THCGeneral.h>

#include "lstm_kernel.h"

float lstm_forward(int hiddenSize, int miniBatch, int seqLength, int numLayers) {
    return forward(hiddenSize, miniBatch, seqLength, numLayers);
}
