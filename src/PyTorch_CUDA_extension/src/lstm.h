int lstm_forward(
              THFloatTensor* h_data,
              THFloatTensor* x_data,
              THFloatTensor* c_data,
              int hiddenSize, int miniBatch, int seqLength, int numLayers);
