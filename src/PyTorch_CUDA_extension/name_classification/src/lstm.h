int lstm_forward(
              THFloatTensor* x_data,
              THFloatTensor* weight,
              THFloatTensor* bias,
              THIntTensor* hiddenSize, 
              THIntTensor* miniBatch, 
              THIntTensor* seqLength, 
              THIntTensor* numLayers);
