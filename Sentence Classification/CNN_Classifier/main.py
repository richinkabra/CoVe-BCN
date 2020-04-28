import os
import argparse
import datetime
import torch
import torchtext.data as data
import torchtext.datasets as datasets
import model
import sys
import torch.autograd as autograd
import torch.nn.functional as F




def dataset_extract(TEXT, LABEL,  **kargs):
    train_data, dev_data, test_data = datasets.SST.splits(TEXT, LABEL, fine_grained=True)
    TEXT.build_vocab(train_data, dev_data, test_data, vectors="glove.6B.300d")
    LABEL.build_vocab(train_data, dev_data, test_data)
    train_iter, dev_iter, test_iter = data.BucketIterator.splits((train_data, dev_data, test_data),batch_sizes=(64,len(dev_data),len(test_data)),**kargs)
    return train_iter, dev_iter, test_iter



def train(train_iter, dev_iter, model, args, test_iter):
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    iterations = 0
    best_acc = 0
    model.train()
    for epoch in range(1, 201):
        print ("Epoch: "+int(epoch))
        for batch in train_iter:
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

                dev_acc = evaluate(dev_iter, model, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    save(model, "./checkpoints", 'best', iterations)

                save(model, "./checkpoints", 'checkpoint', iterations)

        print ("Evaluating on test set...")
        evaluate(test_iter, model, args)

def evaluate(data_iter, model, args):
    model.eval()
    preds, avg_loss = 0, 0
    for batch in data_iter:
        sent, label = batch.text, batch.label
        sent.data.t_(), label.data.sub_(1)
        sent, label = sent.cuda(), label.cuda()

        logit = model(sent)
        loss = F.cross_entropy(logit, label, size_average=False)

        avg_loss += loss.data[0]
        preds += (torch.max(logit, 1)[1].view(label.size()).data == label.data).sum()

    size = len(data_iter.dataset)
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

def main():
    parser = argparse.ArgumentParser(description='Kim (2014) text classificer')


    parser.add_argument('-cove', default=False, help='use cove+glove')
    args = parser.parse_args()

    TEXT = data.Field(lower=True)
    LABEL = data.Field(sequential=False)
    train_iter, dev_iter, test_iter = dataset_extract(TEXT, LABEL, device=-1, repeat=False)

    model = model.CNN_Sent_Class(args,TEXT)
    torch.cuda.set_device(0)
    model = model.cuda()

    try:
        train.train(train_iter, dev_iter, model, args, test_iter)
    except KeyboardInterrupt:
        print('Quitting')


if __name__ == '__main__':
    main()
