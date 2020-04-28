import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


import bcolz
import numpy as np
import pickle
from cove import MTLSTM

class CNN_Sent_Class(nn.Module):

    def __init__(self, args, TEXT):
        super(CNN_Text, self).__init__()
        self.args = args
        vocab = TEXT.vocab
        self.cove = args.cove
        self.embed = nn.Embedding(len(vocab), 300)
        self.embed.weight.data.copy_(vocab.vectors)
        if args.cove:
            self.convs1 = nn.ModuleList([nn.Conv2d(1, 100, (K, 1500)) for K in [3,4,5]])
        else:
            self.convs1 = nn.ModuleList([nn.Conv2d(1, 100, (K, 300)) for K in [3,4,5]])

        self.dropout = nn.Dropout(0.3)
        self.fully_connected = nn.Linear(300, 5)





    def forward(self, x):

        x = self.embed(x)



        if (self.cove):
            outputs_both_layer_cove_with_glove = MTLSTM(n_vocab=None, vectors=None, layer0=True, residual_embeddings=True)
            outputs_both_layer_cove_with_glove.cuda()
            x = outputs_both_layer_cove_with_glove(x,[x.shape[1]]*x.shape[0])

        x = x.unsqueeze(1)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]

        x = torch.cat(x, 1)

        x = self.dropout(x)

        output = self.fully_connected(x)

        return output
