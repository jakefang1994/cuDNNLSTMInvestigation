int lstm_forward(
              THCudaTensor* h_data,
              THCudaTensor* x_data,
              THCudaTensor* c_data,
              int hiddenSize, int miniBatch, int seqLength, int numLayers);
