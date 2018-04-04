#include <TH.h>
#include <THC.h>
#include <THCGeneral.h>
#include <time.h>

#include "lstm_kernel.h"

extern THCState* state;

int lstm_forward(
    THFloatTensor* h_data,
    THFloatTensor* x_data,
    THFloatTensor* c_data,
    THIntTensor* hiddenSize, 
    THIntTensor* miniBatch, 
    THIntTensor* seqLength, 
    THIntTensor* numLayers) {

	clock_t start,end;
	float e_time;
	start = clock();
	
    forward(state, h_data, x_data, c_data, 
            hiddenSize, miniBatch, seqLength, numLayers);

	end = clock();
	e_time = ((float)(end - start)) / CLOCKS_PER_SEC;
	printf("C wrapper time:\t%f\n", e_time);

    return 1;
}
