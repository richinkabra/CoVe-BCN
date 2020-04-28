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


class CNN_Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):

        encoder_conved, encoder_combined = self.encoder(src)

        output, attention = self.decoder(trg, encoder_conved, encoder_combined)

        return output, attention
