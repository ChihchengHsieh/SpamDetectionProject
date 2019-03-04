import torch
import torch.nn as nn
# For the use of padding
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm
from torch.utils.data import DataLoader
import torch.utils.data as Data
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import gensim
import os
from nltk.corpus import wordnet as wn  # For WordNet
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
from sklearn.model_selection import train_test_split
from collections import defaultdict
import time
import pandas as pd

%matplotlib inline

# Import Data
# Padding the incomming Data
# Clip the Grad of LSTM
# Make h0 and C0 trainable

device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')


class args(object):

    # Data

    dataset_path = ""  # load a dataset and setting
    vocab_size = 2000
    seq_len = 20

    # Arch

    usingPretrainedEmbedding = False
    if usingPretrainedEmbedding:
        embedding_dim = 300
    else:
        embedding_dim = 500

    hidden = 32

    # Training params
    batch_size = 5
    L2 = 0
    threshold = 0.5
    lr = 1e-3
    epochs = 3

    # If using Adam
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_weight_decay = 0.01

    # Logging the Training
    log_freq = 10
    model_save_freq = 1
    model_name = 'SSCL'
    model_path = './' + model_name + '/Model/'
    log_path = './' + model_name + '/Log/'


# Create the path for saving model and the log
if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)

if not os.path.exists(args.log_path):
    os.makedirs(args.log_path)


class Constants():
    ''' The Constants for the text '''
    PAD = 1
    UNK = 0
    SOS = 2
    EOS = 3

    PAD_WORD = '<blank>'
    UNK_WORD = '<UNK>'
    SOS_WORD = '<SOS>'
    EOS_WORD = '<EOS>'


# Load Google's pre-trained Word2Vec model.
# model = gensim.models.KeyedVectors.load_word2vec_format(
#     './GoogleNews-vectors-negative300.bin', binary=True)

# word2vector = torch.FloatTensor(model.vectors)


'''
# import the dataset
# Preprocessing the data
# Split the dataset to (traninig, test, validation)
'''


class SSCL(nn.Module):

    ''' The Model from paper '''

    def __init__(self,):
        super(SSCL, self).__init__()

        if args.usingPretrainedEmbedding:
            self.embed = nn.Embedding.from_pretrained(word2vector)
        else:
            self.embed = nn.Embedding(
                args.vocab_size, args.embedding_dim, Constants.PAD)

        self.cnn = nn.Sequential(
            nn.Conv1d(args.embedding_dim, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

        self.rnn = nn.LSTM(64, 128, batch_first=True)

        self.out_net = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.apply(self.weight_init)

    def forward(self, input):

        emb_out = self.embed(input).transpose(1, 2)

        out = self.cnn(emb_out).transpose(1, 2)

        out = self.rnn(out)[0][:, -1, :]

        out = self.out_net(out)

        return out

    def weight_init(self, m):

        if type(m) in [nn.Conv2d, nn.ConvTranspose2d, nn.Linear]:
            nn.init.kaiming_normal_(m.weight, 0.2, nonlinearity='leaky_relu')
        elif type(m) in [nn.LSTM]:
            for name, value in m.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_normal_(value.data)
                if 'bias'in name:
                    value.data.normal_()


# We can put all the lost function and the optimization into the args

class Trainer(nn.Module):

    def __init__(self,):
        super(Trainer, self).__init__()
        self.SSCL = SSCL()
        self.optim = optim.Adagrad(
            self.SSCL.parameters(), lr=args.lr, weight_decay=args.L2)
        self.Loss = nn.BCELoss()
        self.hist = defaultdict(list)

    def forward(self, input, label):

        self.pred = self.SSCL(input)

        loss = self.Loss(self.pred, label)

        accuracy = torch.mean(
            ((self.pred > args.threshold) == label.byte()).float())

        return loss, accuracy

    def train_step(self, input, label):

        self.optim.zero_grad()

        loss, accuracy = self.forward(input, label)

        self.train_hist["Loss"].append(loss.item())
        self.train_hist["Accuracy"].append(accuracy.item())

        self.loss.backward()
        self.optim.step()

    def test_step(self, input, label, validation=True):

        # Not Updating the weight

        loss, accuracy = self.forward(input, label)

        if validation:
            self.train_hist["V_Loss"].append(loss.item())
            self.train_hist["V_Accuracy"].append(accuracy.item())
        else:
            self.train_hist["T_Loss"].append(loss.item())
            self.train_hist["T_Accuracy"].append(accuracy.item())

# input can be = (B, L)


test_input = torch.randint(0, args.vocab_size, (args.batch_size, 15))

test_lable = torch.ones(args.batch_size, 1)

T = Trainer()

T(test_input, test_lable)

SSCL()(test_input)
