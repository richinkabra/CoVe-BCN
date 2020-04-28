import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator
import spacy
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
from docopt import docopt
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from vocab import Vocab, VocabEntry


class LSTM_Decoder(nn.Module):
    def __init__(self, vocabulary, embedding_size, hidden_size, dropout_rate=0.3):
        super().__init__()

        self.dropout_rate = dropout_rate

        self.dropout = nn.Dropout(self.dropout_rate)
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size


        self.vocabulary = vocabulary
        self.decoder_lstm_input = embedding_size + hidden_size
        vocabulary_size = len(vocabulary.tgt)
        self.W1 = nn.Linear(2*hidden_size, hidden_size, bias=False)

        self.W2 = nn.Linear(((2*hidden_size)+hidden_size), hidden_size, bias=False)
        self.embedding_layer_target = nn.Embedding(vocabulary_size, embedding_size, padding_idx=vocabulary.tgt['<pad>'])
        self.lstm_cell = nn.LSTMCell(self.decoder_lstm_input, hidden_size)
        # self.lstm_cell2 = nn.LSTMCell(self.decoder_lstm_input, hidden_size)

    def forward(self, h_enc: torch.Tensor, mask: torch.Tensor, decoder_init_vec: Tuple[torch.Tensor, torch.Tensor], sentences: torch.Tensor) -> torch.Tensor:

        batch_size = h_enc.size(0)

        h_til_tmin1 = torch.zeros(batch_size, self.hidden_size, device="cuda:0")

        tgt_word_embeds = self.embedding_layer_target(sentences)

        h_dec_tmin1 = decoder_init_vec

        h_til = []

        for z_tmin1 in tgt_word_embeds.split(split_size=1):
            z_tmin1 = z_tmin1.squeeze(0)

            h_dec_t, cell_t = self.lstm_cell((torch.cat([z_tmin1, h_til_tmin1], dim=-1)), h_dec_tmin1)
            # hc_1 = self.lstm_cell((torch.cat([z_tmin1, h_til_tmin1], dim=-1)), h_dec_tmin1)
            # h_1,c_1 = hc_1
            # hc_2 = self.lstm_cell2(h_1,hc_2)
            # h_dec_t,cell_t = hc_2
            interim_eqn = torch.bmm(self.W1(h_enc), h_dec_t.unsqueeze(2)).squeeze(2)

            if mask is not None:
                interim_eqn.data.masked_fill_(mask.byte(), -float('inf'))

            alpha_t = F.softmax(interim_eqn, dim=-1)

            att_view = (interim_eqn.size(0), 1, interim_eqn.size(1))

            h_til_t = torch.tanh(self.W2(torch.cat([h_dec_t, (torch.bmm(alpha_t.view(*att_view), h_enc).squeeze(1))], 1)))
            h_til_t = self.dropout(h_til_t)

            h_til_tmin1 = h_til_t
            h_dec_tmin1 = h_dec_t, cell_t
            h_til.append(h_til_t)

        h_til = torch.stack(h_til)

        return h_til
