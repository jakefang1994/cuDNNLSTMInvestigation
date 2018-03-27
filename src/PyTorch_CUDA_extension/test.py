import torch
from lstm import lstm

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

    def forward(self, hiddenSize, miniBatch, seqLength, numLayers):
        return lstm().forward(hiddenSize, miniBatch, seqLength, numLayers)

net = Network().cuda()

hidden_size = torch.autograd.Variable(torch.IntTensor([512]))
mini_batch = torch.autograd.Variable(torch.IntTensor([64]))
seq_length = torch.autograd.Variable(torch.IntTensor([100]))
num_layer = torch.autograd.Variable(torch.IntTensor([4]))

time = net.forward(hidden_size, mini_batch, seq_length, num_layer)

print("Time: ", time)
