#ifdef __cplusplus
	extern "C" {
#endif

void forward(THCState* state,
              THFloatTensor* h_data_cpu,
              THFloatTensor* x_data_cpu,
              THFloatTensor* c_data_cpu,
              THIntTensor* _hiddenSize, 
              THIntTensor* _miniBatch, 
              THIntTensor* _seqLength, 
              THIntTensor* _numLayers);

#ifdef __cplusplus
	}
#endif
