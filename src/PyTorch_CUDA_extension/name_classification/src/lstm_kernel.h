#ifdef __cplusplus
	extern "C" {
#endif

void forward(THCState* state,
              THFloatTensor* x_data_cpu,
              THFloatTensor* weight_cpu,
              THFloatTensor* bias_cpu,
              THIntTensor* _hiddenSize, 
              THIntTensor* _miniBatch, 
              THIntTensor* _seqLength, 
              THIntTensor* _numLayers);

#ifdef __cplusplus
	}
#endif
