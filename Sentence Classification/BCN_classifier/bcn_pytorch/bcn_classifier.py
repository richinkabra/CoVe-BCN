import torch
import torch.nn as nn
import torch.nn.functional as F
import timeit
import numpy as np
import torchtext.data as data
import torchtext.datasets as datasets
import os
import argparse
import datetime
import sys
import torch.autograd as autograd
from maxout import Maxout

class BCN(nn.Module):
  """ Biattentive Classification Network
      Uses BiLSTM Encoder and Integrator to encode input sequences and compute probability distribution over them
  """
  def __init__( self, train_iter, dev_iter, test_iter, n_classes, max_seq_len, encoding_len, bilstm_n_hidden_units, **hyperparams):
    super().__init__()
    self.hyperparams = hyperparams
    self.max_words = max_seq_len
    self.encoding_len = encoding_len
    self.n_hidden_units = bilstm_n_hidden_units
    self.train_iter = train_iter
    self.dev_iter = dev_iter
    self.test_iter = test_iter   

  def create_bcn_model(self):
    print(self.hyperparams.keys())
    batch_size = self.hyperparams['BATCH_SIZE']
    dropout = self.hyperparams['DROPOUT_RATIO']
    l_rate = self.hyperparams['ADAM_ALPHA']
    maxout_pooling = self.hyperparams['MAXOUT_POOL_DIMENSION']
    w_x =  torch.empty((batch_size,self.max_words,self.encoding_len), dtype=torch.float32)
    w_y =  torch.empty((batch_size,self.max_words,self.encoding_len), dtype=torch.float32)
    
    # Passing the two sequences through ReLU Feedforward Layer
    self.linear = nn.Linear((batch_size*self.max_words*self.encoding_len), (batch_size*self.max_words*self.encoding_len))

    # biLSTM to process sequences and obtain task-specific representations
    self.biLSTM_x1 = nn.LSTM(input_size = encoding_len, hidden_size = n_hidden_units, bidirectional = True)
    self.biLSTM_y1 = nn.LSTM(input_size = encoding_len, hidden_size = n_hidden_units, bidirectional = True)
    
    # # Pass the concatenated vectors through a biLSTM
    self.biLSTM_x2 = nn.LSTM(input_size = n_hidden_units*3, hidden_size = n_hidden_units*2, bidirectional = True)
    self.biLSTM_y2 = nn.LSTM(input_size = n_hidden_units*3, hidden_size = n_hidden_units*2, bidirectional = True)

    relu_x = F.relu(self.linear(w_x))
    relu_y = F.relu(self.linear(w_y))

    X = self.biLSTM_x1(relu_x)
    Y = self.biLSTM_y1(relu_y)

    # Affinity Matrix
    A = torch.matmul(X,Y.permute(0,2,1))

    # Extract Attention Weights
    A_x = F.softmax(A)
    A_y = F.softmax(A.permute(0,2,1))

    # Compute Context Summaries
    C_x = torch.matmul(A_x.permute(0,2,1),X)
    C_y = torch.matmul(A_y.permute(0,2,1),Y)

    # Integrate conditioning information
    x_concat = torch.cat((X, X - C_y, X*C_y), 1)
    # Masking to take care of different length sequences here?
    y_concat = torch.cat((Y, Y - C_x, Y*C_x), 1)
    # Masking to take care of different length sequences here?

    # Pass the concatenated vectors through a biLSTM
    X_y = self.biLSTM_x2(x_concat)
    Y_x = self.biLSTM_y2(y_concat)

    # Self attentive Pooling
    
    # Write like equations
    v1 = torch.randn(batch_size, encoding_len*2, 1)
    d1 = torch.randn(batch_size,self.max_words, 1)
    beta_x = F.softmax(torch.matmul(X_y,v1) + d1)

    v2 = torch.randn(batch_size, encoding_len*2, 1)
    d2 = torch.randn(batch_size,self.max_words, 1)
    beta_y = F.softmax(torch.matmul(Y_x,v2) + d2)

    x_self = torch.matmul(X_y.permute(0,2,1),beta_x)
    y_self = torch.matmul(Y_x.permute(0,2,1),beta_y)

    x_maxpool = torch.nn.MaxPool2d(3)(X_y)
    x_avgpool = torch.nn.AvgPool2d(3)(X_y)
    x_minpool = -torch.nn.MaxPool2d(3)(-X_y)

    y_maxpool = torch.nn.MaxPool2d(3)(Y_x)
    y_avgpool = torch.nn.AvgPool2d(3)(Y_x)
    y_minpool = -torch.nn.MaxPool2d(3)(-Y_x)

    x_pool = torch.cat((x_maxpool, x_avgpool, x_minpool, x_self), 0)
    y_pool = torch.cat((y_maxpool, y_avgpool, y_minpool, y_self), 0)

    xy_joined =  torch.cat((x_pool,y_pool),0) 

    # Maxout Layers
    # joined.shape, (self.params['batch_size'], self.params['bilstm_integrate_n_hidden']*2*4*2)
    maxout1_dim = (n_hidden_units*2*4*2)
    dp1 = nn.Dropout(self.dropout_ratio)(xy_joined)
    bn1 = nn.BatchNorm2d(num_features = xy_joined.shape[1], eps = batch_norm_eps, momentum = batch_norm_m, affine = True, track_running_stats = True)(dp1)
    mo1 = Maxout(nn.Linear(bn1.shape,maxout1_dim)(bn1))

    maxout2_dim = maxout1_dim/self.maxout_dim_red#(n_hidden_units,2,4,2)
    dp2 = nn.Dropout(self.dropout_ratio)(mo1)
    bn2 = nn.BatchNorm2d(num_features = dp2.shape[1], eps = batch_norm_eps, momentum = batch_norm_m, affine = True, track_running_stats = True)(dp2)
    mo2 = Maxout(nn.Linear(bn2.shape,(maxout2_dim,maxout2_dim))(bn2))

    maxout3_dim = maxout2_dim/2#(n_hidden_units,2,4,2)
    dp3 = nn.Dropout(self.dropout_ratio)(mo2)
    output = F.softmax(nn.Linear(bn3.shape,(maxout3_dim,self.n_classes))(dp3))

    return output

  def train_bcn_model(self, model):
      n_epochs = self.hyperparams['N_EPOCHS']
      l_rate = self.hyperparams['ADAM_ALPHA']
      optimizer = torch.optim.Adam(model.parameters(), lr = l_rate)
      iterations = 0
      best_acc = 0
      for epoch in range(1, n_epochs):
          print ("Epoch: "+int(epoch))
          for batch in self.train_iter:
              sent, label = batch.text, batch.label
              sent.data.t_(), label.data.sub_(1)
              sent, label = sent.cuda(), label.cuda()
              optimizer.zero_grad()
              logit = model(sent)
              loss = F.cross_entropy(logit, label)
              loss.backward()
              optimizer.step()
              iterations = iterations + 1
              if iterations % 100 == 0:
                  print ("Training Loss: "+ float(loss.data[0]))
                  dev_acc = evaluate_bcn_model(self.dev_iter)
                  if dev_acc > best_acc:
                      best_acc = dev_acc
                      save(model, "./checkpoints", 'best', iterations)
                  save(model, "./checkpoints", 'checkpoint', iterations)

  
  def evaluate_bcn_model(self, model):
      model.eval()
      preds, avg_loss = 0, 0
      for batch in self.data_iter:
          sent, label = batch.text, batch.label
          sent.data.t_(), label.data.sub_(1)
          sent, label = sent.cuda(), label.cuda()

          logit = model(sent)
          loss = F.cross_entropy(logit, label, size_average=False)

          avg_loss += loss.data[0]
          preds += (torch.max(logit, 1)[1].view(label.size()).data == label.data).sum()

      size = len(self.data_iter.dataset)
      avg_loss /= size
      accuracy = 100.0 * preds/size
      print ("Test loss: "+ float(avg_loss))
      print ("Test accuracy: "+ int(accuracy))

      return accuracy

  def save(model, save_dir, save_prefix, iterations):
      if not os.path.isdir(save_dir):
          os.makedirs(save_dir)
      save_prefix = os.path.join(save_dir, save_prefix)
      save_path = '{}_iterations_{}.pt'.format(save_prefix, iterations)
      torch.save(model.state_dict(), save_path)
