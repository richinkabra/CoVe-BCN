import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator
import numpy as np
import random
from torchtext.data.metrics import bleu_score
import math
import pickle
import sys
import time
from typing import List, Tuple, Dict, Set, Union
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from vocab import Vocab, VocabEntry
import bcolz



class LSTM_Encoder(nn.Module):
    def __init__(self, vocabulary, embedding_size, hidden_output_size):
        super().__init__()

        vectors = bcolz.open('./glove.84B.300.dat')[:]
        words = pickle.load(open('./84B.300_words.pkl', 'rb'))
        word2idx = pickle.load(open('./84B.300_idx.pkl', 'rb'))

        vocabulary_size = len(vocabulary.src)

        glove = {w: vectors[word2idx[w]] for w in words}

        matrix_len = vocabulary_size
        weights_matrix = np.zeros((matrix_len, 300))
        for word,i in vocabulary.src.word2id.items():

            try:
                weights_matrix[i] = glove[word]
            except KeyError:
                weights_matrix[i] = np.random.normal(scale=0.5, size=(embedding_size, ))

        self.embedding_size = embedding_size
        self.hidden_output_size = hidden_output_size
        self.vocabulary = vocabulary

        self.dec_initial_state = nn.Linear(hidden_output_size * 2, hidden_output_size)
        self.embedding_layer = nn.Embedding(vocabulary_size, embedding_size, padding_idx=vocabulary.src['<pad>'])
        self.embedding_layer.weight.data.copy_(torch.from_numpy(weights_matrix))
        self.lstm_layer_1 = nn.LSTM(embedding_size, hidden_output_size, bidirectional=True,num_layers=2)


    def forward(self, sentences: torch.Tensor, lengths: List[int]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        source_lang, (last_state, last_cell) = self.lstm_layer_1(pack_padded_sequence((self.embedding_layer(sentences)), lengths))
        source_lang, _ = pad_packed_sequence(source_lang)
        di = self.dec_initial_state(torch.cat([last_cell[0], last_cell[1]], dim=1))
        dec_init_state = torch.tanh(di)

        return (source_lang.permute(1, 0, 2)), (dec_init_state, di)
