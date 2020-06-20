import pickle
from torch.utils.data import Dataset


class instance(object):
    def __init__(self, seq, tag):
        self.seq = seq
        self.tag = tag
        self.leng = len(seq)
        assert len(seq) == len(tag)


class dataset(Dataset):
    def __init__(self, name):
        self.insts = pickle.load(open('./preprocessed/{}_insts.pkl'.format(name), 'rb'))

    def __getitem__(self, index):
        item = {
            "seq": self.insts[index].seq,
            "tag": self.insts[index].tag,
            "leng": self.insts[index].leng
        }
        return item

    def __len__(self):
        return len(self.insts)
