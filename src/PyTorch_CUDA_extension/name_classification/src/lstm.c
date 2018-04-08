#include <TH.h>
#include <THC.h>
#include <THCGeneral.h>
#include <time.h>

#include "lstm_kernel.h"

extern THCState* state;

int lstm_forward(
    THFloatTensor* x_data,
    THFloatTensor* weight,
    THFloatTensor* bias,
    THIntTensor* hiddenSize, 
    THIntTensor* miniBatch, 
    THIntTensor* seqLength, 
    THIntTensor* numLayers) {

	// clock_t start,end;
	// float e_time;
	// start = clock();
	
    forward(state, x_data, weight, bias, 
            hiddenSize, miniBatch, seqLength, numLayers);

	// end = clock();
	// e_time = ((float)(end - start)) / CLOCKS_PER_SEC;
	// printf("C wrapper time:\t%f\n", e_time);

    return 1;
}
