import sys
import time
import torch
from lstm import lstm

# two networks share hyperparameters:
argv = sys.argv
if len(argv) == 5:
    hidden_size = int(argv[1])
    mini_batch = int(argv[2])
    seq_length = int(argv[3])
    num_layer = int(argv[4])
elif len(argv) == 1:
    hidden_size = 512
    mini_batch = 64
    seq_length = 100
    num_layer = 4
else:
    print("Usage: python test.py <hiddenSize> <miniBatch> <seqLength> <numLayers>")
    raise NotImplementedError()

hidden_size_tensor = torch.autograd.Variable(torch.IntTensor([hidden_size]))
mini_batch_tensor = torch.autograd.Variable(torch.IntTensor([mini_batch]))
seq_length_tensor = torch.autograd.Variable(torch.IntTensor([seq_length]))
num_layer_tensor = torch.autograd.Variable(torch.IntTensor([num_layer]))


class Custom_Network(torch.nn.Module):
    def __init__(self):
        super(Custom_Network, self).__init__()

    def forward(self, hiddenSize, miniBatch, seqLength, numLayers):
        # random inputs
        num_elements = hidden_size * mini_batch
        self.x_data = torch.randn(seq_length*(num_layer+1)*num_elements).cuda()
        self.h_data = torch.randn((seq_length+1)*num_layer*num_elements).cuda()
        self.c_data = torch.randn((seq_length+1)*num_layer*num_elements).cuda()

        lstm().forward(self.h_data, self.x_data, self.c_data, hiddenSize, miniBatch, seqLength, numLayers)
        
class Official_Network(torch.nn.Module):
    def __init__(self, hiddenSize, miniBatch, seqLength, numLayers):
        super(Official_Network, self).__init__()
        self.hiddenSize = hiddenSize
        self.miniBatch = miniBatch
        self.seqLength = seqLength
        self.numLayers = numLayers
        self.lstm = torch.nn.LSTM(hiddenSize, hiddenSize, numLayers, batch_first=False)

    def forward(self, x):
        self.h_data = torch.autograd.Variable(torch.randn(self.numLayers, self.miniBatch, self.hiddenSize)).cuda()
        self.c_data = torch.autograd.Variable(torch.randn(self.numLayers, self.miniBatch, self.hiddenSize)).cuda()

        out, _ = self.lstm(x, (self.h_data, self.c_data))


net = Custom_Network().cuda()
start_time = time.time()
net.forward(hidden_size, mini_batch, seq_length, num_layer)
elapsed_time = time.time() - start_time
print("Custom time:\t%f seconds"%(elapsed_time))

net = Official_Network(hidden_size, mini_batch, seq_length, num_layer).cuda()
start_time = time.time()
x_data = torch.autograd.Variable(torch.randn(seq_length, mini_batch, hidden_size)).cuda()
net.forward(x_data)
elapsed_time = time.time() - start_time
print("Official time:\t%f seconds"%(elapsed_time))
