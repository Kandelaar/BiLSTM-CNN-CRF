from train import trainer
from dataset import dataset
from helper import *
import json


if __name__ == '__main__':
    train_data = dataset('train')
    valid_data = dataset('valid')
    test_data = dataset('test')
    vocab = json.load(open('./preprocessed/vocab.json', 'r'))
    char_vocab = json.load(open('./preprocessed/char_vocab.json', 'r'))
    Trainer = trainer(train_data, valid_data, test_data,
                      vocab, len(char_vocab), embedding_dim=100, char_embedding_dim=30, hidden_dim=100,
                      window_size=3, filter_num=30)
    # Trainer.eval(0)
    Trainer.train()
