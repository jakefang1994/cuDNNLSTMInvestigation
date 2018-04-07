import sys
import time
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from lstm import lstm
import numpy as np

#-------- load hyperparameters --------
# two networks share hyperparameters:
argv = sys.argv
if len(argv) == 5:
    hidden_size = int(argv[1])
    mini_batch = int(argv[2])
    seq_length = int(argv[3])
    num_layer = int(argv[4])
elif len(argv) == 1:
    hidden_size = 28
    mini_batch = 100
    seq_length = 28
    num_layer = 2
else:
    print("Usage: python test.py <hiddenSize> <miniBatch> <seqLength> <numLayers>")
    raise NotImplementedError()

num_elements = hidden_size * mini_batch
print("hiddenSize %d, miniBatch %d, seqLength %d, numLayers %d"%(hidden_size, mini_batch, seq_length, num_layer))

hidden_size_tensor = torch.autograd.Variable(torch.IntTensor([hidden_size]), requires_grad=False)
mini_batch_tensor = torch.autograd.Variable(torch.IntTensor([mini_batch]), requires_grad=False)
seq_length_tensor = torch.autograd.Variable(torch.IntTensor([seq_length]), requires_grad=False)
num_layer_tensor = torch.autograd.Variable(torch.IntTensor([num_layer]), requires_grad=False)

#-------- load test images --------
input_size = hidden_size
num_classes = 10
# MNIST Dataset
test_dataset = dsets.MNIST(root='./data/',
                           train=False,
                           transform=transforms.ToTensor(),
                           download=True)
# Data Loader (Input Pipeline)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=mini_batch,
                                          shuffle=False)

#-------- load parameters --------
dict = torch.load("rnn.pkl")
# torch.Size([2, 2, 112, 28])
weight = torch.cat((torch.cat((dict["lstm.weight_ih_l0"].unsqueeze(0).unsqueeze(0),
                               dict["lstm.weight_hh_l0"].unsqueeze(0).unsqueeze(0)), 1),
                    torch.cat((dict["lstm.weight_ih_l1"].unsqueeze(0).unsqueeze(0),
                               dict["lstm.weight_hh_l1"].unsqueeze(0).unsqueeze(0)), 1)),0)
weight_1d = torch.autograd.Variable(np.reshape(weight, (-1))) # torch.Size([12544])
# torch.Size([2, 2, 112])
bias = torch.cat((torch.cat((dict["lstm.bias_ih_l0"].unsqueeze(0).unsqueeze(0),
                             dict["lstm.bias_hh_l0"].unsqueeze(0).unsqueeze(0)), 1),
                  torch.cat((dict["lstm.bias_ih_l1"].unsqueeze(0).unsqueeze(0),
                             dict["lstm.bias_hh_l1"].unsqueeze(0).unsqueeze(0)), 1)),0)
bias_1d = torch.autograd.Variable(np.reshape(bias, (-1))) # torch.Size([448])

#-------- models --------
class Custom_Network(torch.nn.Module):
    def __init__(self):
        super(Custom_Network, self).__init__()
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x_data, weight, bias, hiddenSize, miniBatch, seqLength, numLayers):

        # start_t = time.time()

        out = lstm()(x_data, weight, bias, hiddenSize, miniBatch, seqLength, numLayers)

        # elapsed_t = time.time() - start_t
        # print("Module wrapper time:\t%f seconds"%(elapsed_t))

        out = out.view(mini_batch, hidden_size).cuda()
        # print(out)
        out = self.fc(out)
        # print(np.shape(out))

        return out

class Official_Network(torch.nn.Module):
    def __init__(self, hiddenSize, miniBatch, seqLength, numLayers):
        super(Official_Network, self).__init__()
        self.hiddenSize = hiddenSize
        self.miniBatch = miniBatch
        self.seqLength = seqLength
        self.numLayers = numLayers
        self.lstm = torch.nn.LSTM(hiddenSize, hiddenSize, numLayers, batch_first=False)
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # print(x[0])
        self.h_data = torch.autograd.Variable(torch.zeros(self.numLayers, self.miniBatch, self.hiddenSize).cuda())
        self.c_data = torch.autograd.Variable(torch.zeros(self.numLayers, self.miniBatch, self.hiddenSize).cuda())

        out, _ = self.lstm(x, (self.h_data, self.c_data)) # out: seqLength * miniBatch * hiddenSize
        # print(out[1])
        out = self.fc(out[-1, :, :])
        return out

print("Custom LSTM:")
c_net = Custom_Network().cuda()
net_dict = c_net.state_dict()
pretrained_dict = {k: v for k, v in dict.items() if k in net_dict}
net_dict.update(pretrained_dict)
c_net.load_state_dict(pretrained_dict)

correct = 0
total = 0
elapsed_time = 0
for images, labels in test_loader:
    start_time = time.time()
    images = images.numpy().reshape([mini_batch, seq_length, input_size]).transpose(1, 0, 2)
    images = images.reshape(-1)
    images = torch.autograd.Variable(torch.from_numpy(images))

    outputs = c_net(images, weight_1d, bias_1d, hidden_size_tensor, mini_batch_tensor, seq_length_tensor, num_layer_tensor)
    # print(outputs)

    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()
    end_time = time.time()
    # print(end_time - start_time)
    elapsed_time += end_time - start_time
    # print("Custom time:\t%f seconds"%(elapsed_time))
print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))
print("Average time: %f" % (elapsed_time / (10000 / mini_batch)))


print("----------------")
print("Official LSTM:")
o_net = Official_Network(hidden_size, mini_batch, seq_length, num_layer).cuda()
o_net.load_state_dict(dict)
correct = 0
total = 0
elapsed_time = 0
for images, labels in test_loader:
    start_time = time.time()
    images = images.numpy().reshape([mini_batch, seq_length, input_size]).transpose(1, 0, 2)
    images = torch.autograd.Variable(torch.from_numpy(images)).cuda()

    outputs = o_net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()
    end_time = time.time()
    # print(end_time - start_time)
    elapsed_time += end_time - start_time
    # print("Official time:\t%f seconds"%(elapsed_time))
print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))
print("Average time: %f" % (elapsed_time / (10000 / mini_batch)))
