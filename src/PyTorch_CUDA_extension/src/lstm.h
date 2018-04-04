int lstm_forward(
              THFloatTensor* h_data,
              THFloatTensor* x_data,
              THFloatTensor* c_data,
              THIntTensor* hiddenSize, 
              THIntTensor* miniBatch, 
              THIntTensor* seqLength, 
              THIntTensor* numLayers);
