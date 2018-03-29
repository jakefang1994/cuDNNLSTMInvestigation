import torch
import _ext.lstm

class lstm(torch.autograd.Function):
    def __init__(self):
        super(lstm, self).__init__()

    def forward(self, h_data, x_data, c_data, hiddenSize, miniBatch, seqLength, numLayers):
        if h_data.is_cuda == True and x_data.is_cuda == True and c_data.is_cuda == True:
            _ext.lstm.lstm_forward(h_data, x_data, c_data, hiddenSize, miniBatch, seqLength, numLayers)
        else:
            raise NotImplementedError()