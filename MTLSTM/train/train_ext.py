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
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../modules/'))
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

from torchtext.data.metrics import bleu_score

import Encoder.cnn_encoder
import Decoder.cnn_decoder
import models.CNN_Seq2Seq

import en_core_web_sm
import de_core_news_sm




def evaluate(model, iterator, objective):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for i, batch in enumerate(iterator):

            german = batch.src
            english = batch.trg

            output, _ = model(german, english[:,:-1])


            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            english = english[:,1:].contiguous().view(-1)


            loss = objective(output, english)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)




def main():

    SEED = 1234

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')

    # spacy_de = de_core_news_sm.load()
    # spacy_en = en_core_web_sm.load()

    def helper_german(text):

        return [tok.text for tok in spacy_de.tokenizer(text)]

    def helper_english(text):

        return [tok.text for tok in spacy_en.tokenizer(text)]


    german = Field(tokenize = helper_german,init_token = '<sos>',eos_token = '<eos>',lower = True,batch_first = True)

    english = Field(tokenize = helper_english,init_token = '<sos>',eos_token = '<eos>',lower = True,batch_first = True)

    train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(german, english))


    german.build_vocab(train_data, min_freq = 2)
    english.build_vocab(train_data, min_freq = 2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data),batch_size = 128,device = device)

    input_dimension = len(german.vocab)
    output_dimension = len(english.vocab)
    ignore_pad = english.vocab.stoi[english.pad_token]

    enc = Encoder.cnn_encoder.CNN_Encoder(input_dimension, 300, 600, 5, 0.2, device)
    dec = Decoder.cnn_decoder.CNN_Decoder(output_dimension, 300, 600, 5, 0.2, ignore_pad, device)

    model = models.CNN_Seq2Seq.CNN_Seq2Seq(enc, dec).to(device)

    optimizer = optim.Adam(model.parameters())


    objective = nn.CrossEntropyLoss(ignore_index = ignore_pad)

    best_validation_loss = float('inf')

    for epoch in range(8):

        model.train()

        epoch_loss = 0

        for i, batch in enumerate(train_iterator):

            german = batch.src
            english = batch.trg

            optimizer.zero_grad()

            output, _ = model(german, english[:,:-1])

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            english = english[:,1:].contiguous().view(-1)

            loss = objective(output, english)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

            optimizer.step()

            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_iterator)

        valid_loss = evaluate(model, valid_iterator, objective)

        if valid_loss < best_validation_loss:
            best_validation_loss = valid_loss
            torch.save(model.state_dict(), 'cnn_mtlstm_model.pt')

        print("Epoch: "+ str(epoch+1))
        print("Train Loss: "+ str(train_loss))
        print("Validation Loss: "+ str(valid_loss))


    model.load_state_dict(torch.load('cnn_mtlstm_model.pt'))

    test_loss = evaluate(model, test_iterator, objective)

    print("Test Loss:"+ str(test_loss))




if __name__ == '__main__':
    main()
