import math
import pickle
import sys
import time
from collections import namedtuple
import numpy as np
from typing import List, Tuple, Dict, Set, Union
from docopt import docopt
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from utils import read_corpus, batch_iter
from vocab import Vocab, VocabEntry
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../modules/'))
import Encoder.lstm_encoder
import Decoder.lstm_decoder

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


class RNN_Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, vocabulary, hidden_size , dropout_rate, embedding_size):

        super(RNN_Seq2Seq, self).__init__()


        self.vocabulary = vocabulary
        vocabulary_size = len(vocabulary.tgt)
        self.embedding_size = embedding_size
        self.dropout_rate = dropout_rate
        self.hidden_size = hidden_size

        self.encoder = encoder

        self.decoder = decoder

        self.output_layer = nn.Linear(hidden_size, vocabulary_size, bias=True)



    @property
    def device(self) -> torch.device:
        return self.src_embed.weight.device

    def forward(self, sentences_input: List[List[str]], sentences_output: List[List[str]]) -> torch.Tensor:



        sentences_input_tensor = self.vocabulary.src.to_input_tensor(sentences_input, device="cuda:0")

        lengths = [len(s) for s in sentences_input]

        src_encodings, decoder_init_vec = self.encoder(sentences_input_tensor, lengths)


        mask = torch.zeros(src_encodings.size(0), src_encodings.size(1), dtype=torch.float)
        for i, j in enumerate(lengths):
            mask[i, j:] = 1


        mask = mask.to("cuda:0")

        sentences_output_tensor = self.vocabulary.tgt.to_input_tensor(sentences_output, device="cuda:0")

        h_til = self.decoder(src_encodings, mask, decoder_init_vec, sentences_output_tensor[:-1])

        p_t_log = F.log_softmax(self.output_layer(h_til), dim=-1)


        output_mask = (sentences_output_tensor != self.vocabulary.tgt['<pad>']).float()

        scores = (torch.gather(p_t_log, index=sentences_output_tensor[1:].unsqueeze(-1), dim=-1).squeeze(-1) * output_mask[1:]).sum(dim=0)

        return scores



    @staticmethod
    def load():

        params = torch.load("results_IWSLT/model.bin", map_location=lambda storage, loc: storage)
        vocab = params['vocab']
        encoder = Encoder.lstm_encoder.LSTM_Encoder(vocab, 300, 300)

        decoder = Decoder.lstm_decoder.LSTM_Decoder(vocab, 300, 300, 0.2)

        model = RNN_Seq2Seq(encoder, decoder, vocab, 300, 0.2, 300)

        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(embedding_size=self.embedding_size, hidden_size=self.hidden_size, dropout_rate=self.dropout_rate),
            'vocab': self.vocabulary,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)
