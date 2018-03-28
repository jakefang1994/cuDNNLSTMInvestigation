#ifdef __cplusplus
	extern "C" {
#endif

float forward(THCState* state,
              THCudaTensor* h_data,
              THCudaTensor* x_data,
              THCudaTensor* c_data,
              int hiddenSize, int miniBatch, int seqLength, int numLayers);

#ifdef __cplusplus
	}
#endif
