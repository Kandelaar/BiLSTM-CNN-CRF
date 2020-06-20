import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
import torch.nn.functional as F
import numpy as np
import copy
from param import pad_id, batch_size, tag2idx

device = "cuda:0"


def word_pad(seq, max_word_leng):
    padded = []
    padded.extend(seq)
    for chars in padded:
        while len(chars) < max_word_leng:
            chars.append(pad_id)
    return padded


def sentence_pad(ids, tags, chars, max_seq_leng, max_word_leng, pad_tag=True):
    padded_ids, padded_tags, padded_chars = copy.deepcopy(ids), copy.deepcopy(tags), copy.deepcopy(chars)
    while len(padded_ids) < max_seq_leng:
        padded_ids.append(pad_id)
        if pad_tag is True:
            padded_tags.append(pad_id)
        padded_chars.append([pad_id] * max_word_leng)
    return padded_ids[:max_seq_leng], padded_tags[:max_seq_leng], padded_chars[:max_seq_leng]


def init_emb(embedding, dim):
    scope = np.sqrt(3.0 / dim)
    nn.init.uniform_(embedding, -scope, scope)


'''
def collate_fn_train(insts):
    max_seq_len, max_word_len = 0, 0
    for inst in insts:
        max_seq_len = max(max_seq_len, inst["leng"])
        assert len(inst["char_index"]) == inst["leng"]
        for word in inst["char_index"]:
            max_word_len = max(max_word_len, len(word))
    seq = torch.zeros(len(insts), max_seq_len, device=device, dtype=torch.long)
    tag = torch.zeros(len(insts), max_seq_len, device=device, dtype=torch.long)
    char_index = torch.zeros(len(insts), max_seq_len, max_word_len, device=device, dtype=torch.long)
    leng = torch.zeros(len(insts), device=device)
    for i, inst in enumerate(insts):
        for j in range(inst["leng"]):
            seq[i][j] = inst["seq"][j]
            tag[i][j] = encode(inst["tag"][j])
            for k in range(len(inst["char_index"][j])):
                char_index[i][j][k] = inst["char_index"][j][k]
        leng[i] = inst["leng"]
    return {
        "seq": seq,
        "tag": tag,
        "leng": leng,
        "char_index": char_index
    }


def collate_fn_eval(insts):
    max_seq_len, max_word_len = 0, 0
    for inst in insts:
        max_seq_len = max(max_seq_len, inst["leng"])
        assert len(inst["char_index"]) == inst["leng"]
        for word in inst["char_index"]:
            max_word_len = max(max_word_len, len(word))
    seq = torch.zeros(len(insts), max_seq_len, device=device, dtype=torch.long)
    char_index = torch.zeros(len(insts), max_seq_len, max_word_len, device=device, dtype=torch.long)
    leng = torch.zeros(len(insts), device=device)
    tag = [inst["tag"] for inst in insts]
    for i, inst in enumerate(insts):
        for j in range(inst["leng"]):
            seq[i][j] = inst["seq"][j]
            for k in range(len(inst["char_index"][j])):
                char_index[i][j][k] = inst["char_index"][j][k]
        leng[i] = inst["leng"]
    return {
        "seq": seq,
        "tag": tag,
        "leng": leng,
        "char_index": char_index
    }
'''


def collate_fn_train(insts):
    max_seq_len = 0
    for inst in insts:
        max_seq_len = max(max_seq_len, inst["leng"])
    seq = [inst["seq"] for inst in insts]
    tag = torch.zeros(len(insts), max_seq_len, device=device, dtype=torch.long)
    leng = torch.zeros(len(insts), device=device)
    for i, inst in enumerate(insts):
        for j in range(inst["leng"]):
            tag[i][j] = encode(inst["tag"][j])
        leng[i] = inst["leng"]
    return {
        "seq": seq,
        "tag": tag,
        "leng": leng,
    }


def collate_fn_eval(insts):
    max_seq_len = 0
    for inst in insts:
        max_seq_len = max(max_seq_len, inst["leng"])
    seq = [inst["seq"] for inst in insts]
    leng = torch.zeros(len(insts), device=device)
    tag = [inst["tag"] for inst in insts]
    for i, inst in enumerate(insts):
        leng[i] = inst["leng"]
    return {
        "seq": seq,
        "tag": tag,
        "leng": leng,
    }


def build_dataloader(dataset, is_eval=False):
    if is_eval is True:
        return DataLoader(dataset, sampler=SequentialSampler(dataset), collate_fn=collate_fn_eval)
    return DataLoader(dataset, sampler=RandomSampler(dataset), collate_fn=collate_fn_train)


def decode(idx, idx2tag, mask):
    decoded = [[idx2tag[id] for j, id in enumerate(_) if mask[i, j] == 1] for i, _ in enumerate(idx)]
    return decoded


def encode(tag):
    return tag2idx[tag]
