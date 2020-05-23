# Transfer Learning in NLP - Context Vectors (CoVe)

Codebase to generate contextualized word vectors by training a sequence-to-sequence model based on LSTMs for machine translation (MT) task. The hidden state output of the machine translation model’s encoder can be called CoVe (Context Vectors) and be used to represent useful context-based information about text. To show the improvement in accuracy in downstream sentiment and question classification tasks (SST-2, SST-5, IMDb, TREC-6, and TREC-50 datasets), a Biattentive Classification Network (BCN) is used. The BCN results show that using CoVe has a higher test accuracy than random, GloVe, or character embeddings.

## Primary paper

Learned in Translation: Contextualized Word Vectors: Bryan McCann, James Bradbury, Caiming Xiong, Richard Socher (https://arxiv.org/abs/1708.00107)

## Extension papers

Convolutional Sequence to Sequence Learning: Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, Yann N. Dauphin (https://arxiv.org/abs/1705.03122)

Convolutional Neural Networks for Sentence Classification: Yoon Kim (https://arxiv.org/abs/1408.5882)

Deep contextualized word representations: Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, Luke Zettlemoyer (https://arxiv.org/abs/1802.05365)

# Requirements

blis==0.4.1
catalogue==1.0.0
certifi==2020.4.5.1
chardet==3.0.4
cymem==2.0.3
docopt==0.6.2
idna==2.9
importlib-metadata==1.6.0
murmurhash==1.0.2
nltk==3.4.5
numpy==1.18.2
plac==1.1.3
preshed==3.0.2
requests==2.23.0
sentencepiece==0.1.85
six==1.14.0
spacy==2.2.4
srsly==1.0.2
thinc==7.4.0
torch==1.2.0
torchtext==0.5.0
tqdm==4.45.0
urllib3==1.25.8
wasabi==0.6.0
zipp==3.1.0
