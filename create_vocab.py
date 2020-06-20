import json
import re
import os
from param import PAD_TAG


train_file = open('./data/train.txt')
dev_file = open('./data/valid.txt')
test_file = open('./data/test.txt')
path = './preprocessed'

vocab = {PAD_TAG: 0}
char_vocab = {PAD_TAG: 0}
files = [('train', train_file), ('valid', dev_file), ('test', test_file)]
max_word_len = 0
max_sentence_len = 0

if not os.path.exists(path):
    os.mkdir(path)

for name, file in files:
    data = []
    labels = []
    sentence = []
    for row in file:
        if row == '\n':
            if sentence:
                assert len(sentence) == len(labels)
                max_sentence_len = max(max_sentence_len, len(sentence))
                data.append((sentence, labels))
                sentence = []
                labels = []
        else:
            row = row.strip().split(' ')
            word = re.sub('\d', '0', row[0].lower())
            label = row[-1]
            sentence.append(word)
            labels.append(label)
            if not vocab.__contains__(word):
                vocab[word] = len(vocab)
            max_word_len = max(max_word_len, len(word))
            for char in word:
                if not char_vocab.__contains__(char):
                    char_vocab[char] = len(char_vocab)

with open(os.path.join(path, 'vocab.json'), 'w') as fp:
    json.dump(vocab, fp)

with open(os.path.join(path, 'char_vocab.json'), 'w') as fp:
    json.dump(char_vocab, fp)

print('max word length: ', max_word_len)
print('max sentence length: ', max_sentence_len)



