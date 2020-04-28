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
import math
import pickle
import sys
from collections import namedtuple
from typing import List, Tuple, Dict, Set, Union
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class CNN_Decoder(nn.Module):
    def __init__(self, output_dimension, embedding_dimension, hidden_output_dimension, convolution_layers, dropout, padding_tok, device):

        super().__init__()

        self.padding_tok = padding_tok
        self.device = device
        self.covolution_modules_layer = nn.ModuleList([nn.Conv1d(in_channels = hidden_output_dimension,out_channels = 2 * hidden_output_dimension,kernel_size = 3) for _ in range(convolution_layers)])

        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)

        self.source_emb = nn.Embedding(output_dimension, embedding_dimension)
        self.position_source_emb = nn.Embedding(120, embedding_dimension)

        self.transform_emb_hid = nn.Linear(embedding_dimension, hidden_output_dimension)
        self.transform_hid_emb = nn.Linear(hidden_output_dimension, embedding_dimension)

        self.attention_he = nn.Linear(hidden_output_dimension, embedding_dimension)
        self.attention_eh = nn.Linear(embedding_dimension, hidden_output_dimension)

        self.fc_out = nn.Linear(embedding_dimension, output_dimension)


        self.dropout = nn.Dropout(dropout)


    def forward(self, target, encoder_conved, encoder_combined):


        target_length = target.shape[1]

        embedded = self.dropout(self.source_emb(target) + self.position_source_emb(torch.arange(0, target_length).unsqueeze(0).repeat(target.shape[0], 1).to(self.device)))

        conv_input = (self.transform_emb_hid(embedded)).permute(0, 2, 1)

        hidden_output_dimension = conv_input.shape[1]

        for i, conv in enumerate(self.covolution_modules_layer):

            conv_input = self.dropout(conv_input)

            conved = conv(torch.cat((torch.zeros(conv_input.shape[0],hidden_output_dimension,2).fill_(self.padding_tok).to(self.device), conv_input), dim = 2))

            conved = F.glu(conved, dim = 1)

            conved_emb = self.attention_he(conved.permute(0, 2, 1))

            attention = F.softmax(torch.matmul((conved_emb + embedded) * self.scale, encoder_conved.permute(0, 2, 1)), dim=2)

            attended_encoding = (self.attention_eh(torch.matmul(attention, encoder_combined))).permute(0, 2, 1)

            conved = (conved + attended_encoding) * self.scale

            conved = (conved + conv_input) * self.scale

            conv_input = conved

        conved = self.transform_hid_emb(conved.permute(0, 2, 1))

        output = self.fc_out(self.dropout(conved))


        return output, attention
