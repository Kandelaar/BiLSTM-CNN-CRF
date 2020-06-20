import json
import pickle
import re
from dataset import instance
import os


train_file = open('./data/train.txt')
dev_file = open('./data/valid.txt')
test_file = open('./data/test.txt')
path = './preprocessed'
files = [('train', train_file), ('valid', dev_file), ('test', test_file)]

for name, file in files:
    save_name = os.path.join(path, name + '.pkl')
    insts = []
    seq = []
    tag = []
    for row in file:
        if row == '\n':
            if seq:
                assert len(seq) == len(tag)
                insts.append(instance(seq, tag))
                seq = []
                tag = []
        else:
            row = row.strip().split(' ')
            word = re.sub('\d', '0', row[0].lower())
            label = row[-1]
            seq.append(word)
            tag.append(label)
    if not os.path.exists(path):
        os.mkdir(path)
    with open(os.path.join(path, name + '_insts.pkl'), 'wb') as writer:
        pickle.dump(insts, writer)
