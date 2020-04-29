import torch
import argparse
import torchtext.data as data
import torchtext.datasets as datasets

def dataset_extract(TEXT, LABEL,  **kargs):
    train_data, dev_data, test_data = datasets.SST.splits(TEXT, LABEL, fine_grained=True)
    TEXT.build_vocab(train_data, dev_data, test_data, vectors="glove.6B.300d")
    LABEL.build_vocab(train_data, dev_data, test_data)
    train_iter, dev_iter, test_iter = data.BucketIterator.splits((train_data, dev_data, test_data),batch_sizes=(64,len(dev_data),len(test_data)),**kargs)
    return train_iter, dev_iter, test_iter

def main():

    hyperparameters ={
    'N_EPOCHS': 20,
    'BATCH_SIZE': 32,
    'DROPOUT_RATIO': 0.2,
    'OPTIMIZER': "Adam",
    'ADAM_ALPHA': 0.001,
    'MAXOUT_POOL_DIMENSION': 2}

    print(hyperparameters.keys())

    n_classes = 2   #SST-2 contains two-classes
    max_seq_len = 35
    encoding_len = 300
    bilstm_n_hidden_units = 300
    TEXT = data.Field(lower=True)
    LABEL = data.Field(sequential=False)
    train_iter, dev_iter, test_iter = dataset_extract(TEXT, LABEL, device=-1, repeat=False)

    bcn = BCN(train_iter, dev_iter, test_iter, n_classes, max_seq_len, encoding_len, bilstm_n_hidden_units, **hyperparameters)
    model = bcn.create_bcn_model()
    torch.cuda.set_device(0)
    model = model.cuda()

    # # Training 
    # bcn.train_bcn_model(model)

    # # Evaluating
    # bcn.evaluate_bcn_model(model)


if __name__ == '__main__':
    main()
