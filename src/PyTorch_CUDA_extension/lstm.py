import torch
import _ext.lstm

class lstm(torch.autograd.Function):
    def __init__(self):
        super(lstm, self).__init__()

    def forward(self, hiddenSize, miniBatch, seqLength, numLayers):
        return _ext.lstm.lstm_forward(hiddenSize, miniBatch, seqLength, numLayers)
