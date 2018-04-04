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

num_elements = hidden_size * mini_batch
print("hiddenSize %d, miniBatch %d, seqLength %d, numLayers %d"%(hidden_size, mini_batch, seq_length, num_layer))

hidden_size_tensor = torch.autograd.Variable(torch.IntTensor([hidden_size]), requires_grad=False)
mini_batch_tensor = torch.autograd.Variable(torch.IntTensor([mini_batch]), requires_grad=False)
seq_length_tensor = torch.autograd.Variable(torch.IntTensor([seq_length]), requires_grad=False)
num_layer_tensor = torch.autograd.Variable(torch.IntTensor([num_layer]), requires_grad=False)


class Custom_Network(torch.nn.Module):
    def __init__(self):
        super(Custom_Network, self).__init__()

    def forward(self, hiddenSize, miniBatch, seqLength, numLayers):
        
        start_t = time.time()
        
        # random inputs
        self.x_data = torch.autograd.Variable(torch.randn(seq_length * num_elements))
        self.h_data = torch.autograd.Variable(torch.randn(num_layer * num_elements))
        self.c_data = torch.autograd.Variable(torch.randn(num_layer * num_elements))

        # start_t = time.time()

        out = lstm()(self.h_data, self.x_data, self.c_data, hiddenSize, miniBatch, seqLength, numLayers)

        elapsed_t = time.time() - start_t
        print("Module wrapper time:\t%f seconds"%(elapsed_t))
        
class Official_Network(torch.nn.Module):
    def __init__(self, hiddenSize, miniBatch, seqLength, numLayers):
        super(Official_Network, self).__init__()
        self.hiddenSize = hiddenSize
        self.miniBatch = miniBatch
        self.seqLength = seqLength
        self.numLayers = numLayers
        self.lstm = torch.nn.LSTM(hiddenSize, hiddenSize, numLayers, batch_first=False)

    def forward(self, x):
        self.h_data = torch.autograd.Variable(torch.randn(self.numLayers, self.miniBatch, self.hiddenSize).cuda())
        self.c_data = torch.autograd.Variable(torch.randn(self.numLayers, self.miniBatch, self.hiddenSize).cuda())

        out, _ = self.lstm(x, (self.h_data, self.c_data))


net = Custom_Network().cuda()
start_time = time.time()
net.forward(hidden_size_tensor, mini_batch_tensor, seq_length_tensor, num_layer_tensor)
elapsed_time = time.time() - start_time
print("Custom time:\t%f seconds"%(elapsed_time))

print("----------------")

net = Official_Network(hidden_size, mini_batch, seq_length, num_layer).cuda()
start_time = time.time()
x_data = torch.autograd.Variable(torch.randn(seq_length, mini_batch, hidden_size).cuda())
net.forward(x_data)
elapsed_time = time.time() - start_time
print("Official time:\t%f seconds"%(elapsed_time))
