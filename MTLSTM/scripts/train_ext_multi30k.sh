#!/bin/sh
python3 -m spacy download en
python3 -m spacy download de

python3 ./train/train_ext.py
