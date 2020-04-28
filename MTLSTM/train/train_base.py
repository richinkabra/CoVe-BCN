
import math
import pickle
import sys
import time
from collections import namedtuple
import os
import sys
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__),'../modules/'))
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

import numpy as np
from typing import List, Tuple, Dict, Set, Union
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from vocab import Vocab, VocabEntry

import Encoder.lstm_encoder
import Decoder.lstm_decoder
import models.RNN_Seq2Seq


#  Corpus reader method used as-is from online tutorial on PyTorch
def read_corpus(spath, source):
    sentences = []
    for line in open(spath):
        sentence = line.strip().split(' ')
        if source == 'tgt':
            sentence = ['<s>'] + sentence + ['</s>']
        sentences.append(sentence)

    return sentences


#  Batch iteration generator method used as-is from online tutorial on PyTorch
def batch_iter(data, batch_size):
    batch_num = math.ceil(len(data) / batch_size)
    ind = list(range(len(data)))

    np.random.shuffle(ind)

    for i in range(batch_num):
        indices = ind[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]
        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents



def train(args):
    if args.dataset == "small":
        english_train = read_corpus("./data_multi30k/train.en", source='src')
        german_train = read_corpus("./data_multi30k/train.de", source='tgt')

        english_val = read_corpus("./data_multi30k/val.en", source='src')
        german_val = read_corpus("./data_multi30k/val.de", source='tgt')

        train = list(zip(english_train, german_train))
        val = list(zip(english_val, german_val))

        spath = "./model_multi30k.bin"

        vocab = Vocab.load("./data_multi30k/vocab.json")
    else:
        english_train = read_corpus("./data_IWSLT/train.en", source='src')
        german_train = read_corpus("./data_IWSLT/train.de", source='tgt')

        english_val = read_corpus("./data_IWSLT/valid.en", source='src')
        german_val = read_corpus("./data_IWSLT/valid.de", source='tgt')

        train = list(zip(english_train, german_train))
        val = list(zip(english_val, german_val))

        spath = "./model_IWSLT.bin"

        vocab = Vocab.load("./data_IWSLT/vocab.json")


    # Some of the hyperparameters present in this code may be slightly different from the ones used at the start
    # to obtain the results mentioned in the report. This is because the experiments were carried out on a google
    # cloud vm and a lot of these hyperparameters (learning rate, num of layers, hidden size etc) were played around
    # with in order to obtain more insight into the architecture. As maintaining separate versions of files used for
    # such experiments was tedious (as it mostly involved commenting out 2-3 lines or changing the value of num of layers etc),
    # we upload the codebase which has the exact architecture, but may have slightly different hyperparameters than what
    # we initially used to verify the reproducibility of our implementation.

    encoder = Encoder.lstm_encoder.LSTM_Encoder(vocab, 300, 300)

    decoder = Decoder.lstm_decoder.LSTM_Decoder(vocab, 300, 300, 0.2)

    model = models.RNN_Seq2Seq.RNN_Seq2Seq(encoder, decoder, vocab, 300, 0.2, 300)

    model.train()

    for p in model.parameters():
        p.data.uniform_(-0.1, 0.1)


    device = torch.device("cuda:0")


    model = model.to(device)
    print('Begin training')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.2)
    last_perplexity = 0.0
    train_iter = 0
    epoch = 0
    validation_history = []


    epochs = 4

    while True:
        epoch += 1
        print ("Epoch: "+ str(epoch))
        if (epoch == epochs):
            print('Training Stopped')
            exit(0)
        for english_sentencees, german_sentencees in batch_iter(train, batch_size=128):
            train_iter += 1

            optimizer.zero_grad()

            batch_size = len(english_sentencees)
            bloss = -model(english_sentencees, german_sentencees)
            batch_loss = bloss.sum()

            loss = batch_loss / batch_size
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)

            optimizer.step()

        validation_perplexity = get_perplexity(model, val, batch_size=128)
        neg_ppl = -validation_perplexity

        validation_improved = len(validation_history) == 0 or neg_ppl > max(validation_history)
        validation_history.append(neg_ppl)


        if validation_improved:
            model.save(spath)
            torch.save(optimizer.state_dict(), spath + '.optim')

        if last_perplexity<validation_perplexity:

            lr = (optimizer.param_groups[0]['lr'] * 0.5)

            params = torch.load(spath, map_location=lambda storage, loc: storage)
            model.load_state_dict(params['state_dict'])
            model = model.to(device)

            optimizer.load_state_dict(torch.load(spath + '.optim'))
            print ("Lowering learning rate now...")
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        last_perplexity = validation_perplexity







def get_perplexity(model, data, batch_size=128):
    loss = 0.0
    index = 0.0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for english_sentencees, german_sentencees in batch_iter(data, batch_size):
            loss = -model(english_sentencees, german_sentencees).sum()
            loss += loss.item()
            ind = sum(len(s[1:]) for s in german_sentencees)
            index += ind
        perplexity = np.exp(loss/index)

    if was_training:
        model.train()

    return perplexity


def main():
    parser = argparse.ArgumentParser(description='Train base MTLSTM network with small or medium dataset')
    parser.add_argument('-dataset', default="small", help='specify small or medium dataset to be used')
    args = parser.parse_args()

    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    np.random.seed(1234 * 13 // 7)
    train(args)

if __name__ == '__main__':
    main()
