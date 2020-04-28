import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator
import numpy as np
import random
import math
import time
from torchtext.data.metrics import bleu_score
import pickle
import sys
import time
from collections import namedtuple
import numpy as np
from typing import List, Tuple, Dict, Set, Union
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class CNN_Encoder(nn.Module):
    def __init__(self,input_dimension, embedding_dimension, hidden_output_dimension, convolution_layers, dropout, device):
        super().__init__()


        self.device = device

        self.covolution_modules_layer = nn.ModuleList([nn.Conv1d(in_channels = hidden_output_dimension,out_channels = 2 * hidden_output_dimension,kernel_size = 3,padding = 1) for _ in range(convolution_layers)])

        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)

        self.source_emb = nn.Embedding(input_dimension, embedding_dimension)
        self.position_source_emb = nn.Embedding(120, embedding_dimension)

        self.transform_emb_hid = nn.Linear(embedding_dimension, hidden_output_dimension)
        self.transform_hid_emb = nn.Linear(hidden_output_dimension, embedding_dimension)

        c_list = []

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):

        if (len(list(src.shape))<3):

            batch_size = src.shape[0]
            src_len = src.shape[1]
            embedded_source = self.source_emb(src)
            embedded_position = self.position_source_emb(torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device))

        elif(len(list(src.shape))==3):
            batch_size = src.shape[0]
            src_len = src.shape[1]

            embedded_source = src
            embedded_position = self.position_source_emb(torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device))

        embedded = self.dropout(embedded_source + embedded_position)

        conv_input = (self.transform_emb_hid(embedded)).permute(0, 2, 1)

        for i, conv in enumerate(self.covolution_modules_layer):

            conved = F.glu((conv(self.dropout(conv_input))), dim = 1)

            conved = (conved + conv_input) * self.scale

            conv_input = conved

        conved = self.transform_hid_emb(conved.permute(0, 2, 1))

        return conved, ((conved + embedded) * self.scale)
