#ifdef __cplusplus
	extern "C" {
#endif

void forward(THCState* state,
              THFloatTensor* h_data_cpu,
              THFloatTensor* x_data_cpu,
              THFloatTensor* c_data_cpu,
              int hiddenSize, int miniBatch, int seqLength, int numLayers);

#ifdef __cplusplus
	}
#endif
