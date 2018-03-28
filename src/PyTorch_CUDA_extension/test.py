import sys
import torch
from lstm import lstm

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

    def forward(self, h_data, x_data, c_data, hiddenSize, miniBatch, seqLength, numLayers):
        return lstm().forward(h_data, x_data, c_data, hiddenSize, miniBatch, seqLength, numLayers)

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

# random inputs
num_elements = hidden_size * mini_batch
h_data = torch.randn((seq_length+1)*num_layer*num_elements).cuda()
x_data = torch.randn(seq_length*(num_layer+1)*num_elements).cuda()
c_data = torch.randn((seq_length+1)*num_layer*num_elements).cuda()

net = Network().cuda()

hidden_size_tensor = torch.autograd.Variable(torch.IntTensor([hidden_size]))
mini_batch_tensor = torch.autograd.Variable(torch.IntTensor([mini_batch]))
seq_length_tensor = torch.autograd.Variable(torch.IntTensor([seq_length]))
num_layer_tensor = torch.autograd.Variable(torch.IntTensor([num_layer]))

time = net.forward(h_data, x_data, c_data, hidden_size, mini_batch, seq_length, num_layer)

print("Time: ", time)
