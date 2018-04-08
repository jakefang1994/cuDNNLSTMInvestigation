from __future__ import unicode_literals, print_function, division
from io import open
import glob

import unicodedata
import string

import torch
import torch.nn as nn
from torch.autograd import Variable

import random

import time
import math

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from lstm import lstm
import numpy as np

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


######
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    # category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
    # line_tensor = Variable(lineToTensor(line))
    category_tensor = torch.LongTensor([all_categories.index(category)])
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

def padd_zero_variable_sequence(tensor, pad_sequence_length):
    input_size = tensor.size()[-1]
    sequence_length = tensor.size()[0]
    assert sequence_length <= pad_sequence_length, "out of sequence length, please increase the sequence length"
    # tensor.view([sequence_length, input_size])
    tensor_pad = torch.zeros(pad_sequence_length, input_size)
    for i in range(sequence_length):
        tensor_pad[i] = tensor[i, :,:]

    return tensor_pad


def generate_data(batch_size, sequence_length, input_size):
    max_length = sequence_length
    batch_data = torch.zeros(batch_size, max_length, input_size)
    batch_target = torch.LongTensor(batch_size,1)
    batch_data_length = torch.zeros(batch_size,1)

    category_list = []
    for i in range(batch_size):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        line_tensor = padd_zero_variable_sequence(line_tensor, sequence_length)
        batch_data[i] = line_tensor
        batch_data_length[i] = line_tensor.size()[0]
        batch_target[i] = category_tensor

        category_list.append(category)

    batch_target = batch_target.view(torch.numel(batch_target))
    # Transpose to fit shape in lstm (sequence, batch, input_size)
    batch_data = batch_data.numpy().reshape([batch_size, sequence_length, input_size]).transpose(1, 0, 2)
    batch_data = batch_data.reshape(-1)

    batch_data = Variable(torch.from_numpy(batch_data))
    batch_target = Variable(batch_target)
    return batch_data, batch_target, category_list

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def categoryFromOutput(output):
    batch_size = output.data.shape[0]
    guess_i_list = []
    guess_list = []
    for j in range(batch_size):
        top_n, top_i = output.data[j,:].topk(1) # Tensor out of Variable with .data
        guess_i = top_i[0]
        guess_i_list.append(guess_i)
        guess_list.append(all_categories[guess_i])
    return guess_list, guess_i_list


###### Model
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x_data, weight, bias, hiddenSize, miniBatch, seqLength, numLayers):

        out = lstm()(x_data, weight, bias, hiddenSize, miniBatch, seqLength, numLayers)

        out = out.view(mini_batch, hidden_size).cuda()
        out = self.fc(out)
        return out


def evaluate():
    data, target, category = generate_data(mini_batch, seq_length, input_size)
    output = rnn(data, weight_1d, bias_1d, hidden_size_tensor, mini_batch_tensor, seq_length_tensor, num_layer_tensor)
    return output, target, category


if __name__ == "__main__":

    dict = torch.load("rnn-name.pkl")

    weight = torch.cat((torch.cat((dict["lstm.weight_ih_l0"].unsqueeze(0).unsqueeze(0),
                                dict["lstm.weight_hh_l0"].unsqueeze(0).unsqueeze(0)), 1)),0)
                        # torch.cat((dict["lstm.weight_ih_l1"].unsqueeze(0).unsqueeze(0),
                        #         dict["lstm.weight_hh_l1"].unsqueeze(0).unsqueeze(0)), 1)),0)
    weight_1d = torch.autograd.Variable(np.reshape(weight, (-1)))
    bias = torch.cat((torch.cat((dict["lstm.bias_ih_l0"].unsqueeze(0).unsqueeze(0),
                                dict["lstm.bias_hh_l0"].unsqueeze(0).unsqueeze(0)), 1)),0)
                    # torch.cat((dict["lstm.bias_ih_l1"].unsqueeze(0).unsqueeze(0),
                    #             dict["lstm.bias_hh_l1"].unsqueeze(0).unsqueeze(0)), 1)),0)
    bias_1d = torch.autograd.Variable(np.reshape(bias, (-1)))

    ######## util functions
    def findFiles(path):
        return glob.glob(path)


    print(findFiles('data/names/*.txt'))

    all_letters = string.ascii_letters + " .,;'"
    n_letters = len(all_letters)


    # Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
    def unicodeToAscii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in all_letters
        )


    # Build the category_lines dictionary, a list of names per language
    category_lines = {}
    all_categories = []


    # Read a file and split into lines
    def readLines(filename):
        lines = open(filename, encoding='utf-8').read().strip().split('\n')
        return [unicodeToAscii(line) for line in lines]


    for filename in findFiles('data/names/*.txt'):
        category = filename.split('/')[-1].split('.')[0]
        all_categories.append(category)
        lines = readLines(filename)
        category_lines[category] = lines


    ######## LSTM Configuration
    num_classes = 18
    input_size = 57
    hidden_size = 57
    num_layers = 1

    mini_batch = 1
    seq_length = 20

    hidden_size_tensor = torch.autograd.Variable(torch.IntTensor([hidden_size]), requires_grad=False)
    mini_batch_tensor = torch.autograd.Variable(torch.IntTensor([mini_batch]), requires_grad=False)
    seq_length_tensor = torch.autograd.Variable(torch.IntTensor([seq_length]), requires_grad=False)
    num_layer_tensor = torch.autograd.Variable(torch.IntTensor([num_layers]), requires_grad=False)

    rnn = LSTM().cuda()
    net_dict = rnn.state_dict()
    pretrained_dict = {k: v for k, v in dict.items() if k in net_dict}
    net_dict.update(pretrained_dict)
    rnn.load_state_dict(pretrained_dict)
    

    ############ Test
    confusion = torch.zeros(num_classes, num_classes)
    n_confusion = 10000

    elapsed_time = 0
    for i in range(n_confusion):
        start_time = time.time()
        output, target, category = evaluate()
        guess_list, guess_i_list = categoryFromOutput(output)
        end_time = time.time()
        elapsed_time += end_time - start_time

        category_i_list = []
        for i in range(len(category)):
            category_i_list.append(all_categories.index(category[i]))

        for j in range(len(category_i_list)):
            category_i = category_i_list[j]
            guess_i = guess_i_list[j]
            confusion[category_i][guess_i] += 1
        # category_i = all_categories.index(category)
        # confusion[category_i][guess_i] += 1

    print("Average time: %f" % (elapsed_time / 10000))

    # Normalize by dividing every row by its sum
    for i in range(num_classes):
        confusion[i] = confusion[i] / confusion[i].sum()

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    # fig.savefig('confusion.png')
    plt.show()
