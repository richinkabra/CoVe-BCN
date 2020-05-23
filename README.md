# Transfer Learning in NLP - Context Vectors (CoVe)

Codebase to generate contextualized word vectors by training a sequence-to-sequence model based on LSTMs for machine translation (MT) task. The hidden state output of the machine translation model’s encoder can be called CoVe (Context Vectors) and be used to represent useful context-based information about text. To show the improvement in accuracy in downstream sentiment and question classification tasks (SST-2, SST-5, IMDb, TREC-6, and TREC-50 datasets), a Biattentive Classification Network (BCN) is used. The BCN results show that using CoVe has a higher test accuracy than random, GloVe, or character embeddings.

## Primary paper

Learned in Translation: Contextualized Word Vectors: Bryan McCann, James Bradbury, Caiming Xiong, Richard Socher (https://arxiv.org/abs/1708.00107)

## Extension papers

Convolutional Sequence to Sequence Learning: Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, Yann N. Dauphin (https://arxiv.org/abs/1705.03122)

Convolutional Neural Networks for Sentence Classification: Yoon Kim (https://arxiv.org/abs/1408.5882)

Deep contextualized word representations: Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, Luke Zettlemoyer (https://arxiv.org/abs/1802.05365)
