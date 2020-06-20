from helper import *
from param import *
import os
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from allennlp.modules.elmo import batch_to_ids, Elmo

torch.manual_seed(1)


def log_sum_exp(vec, m_size):
    _, idx = torch.max(vec, dim=1)
    max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)
    return max_score.view(-1, m_size) + torch.log(torch.sum(torch.exp(
        vec - max_score.expand_as(vec)), 1)).view(-1, m_size)


class BiLSTM_CNN_CRF(nn.Module):
    def __init__(self, vocab_size, char_vocab_size, embedding_dim, char_embedding_dim, hidden_dim, window_size, filter_num):
        super(BiLSTM_CNN_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.char_embedding_dim = char_embedding_dim
        self.filter_num = filter_num
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag2idx = tag2idx
        self.tag_num = len(tag2idx)

        # self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_dim, padding_idx=pad_id)
        # self.dropout_char = nn.Dropout(0.1)
        # init_emb(self.char_embedding.weight, char_embedding_dim)
        # self.conv = nn.Conv3d(in_channels=1, out_channels=filter_num,
        #                       kernel_size=(1, window_size, self.char_embedding_dim))

        # self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        # init_emb(self.embedding.weight, embedding_dim)
        # self.dropout = nn.Dropout(0.5)
        self.elmo = Elmo(options_file, weight_file, num_output_representations=1, dropout=0, requires_grad=True, do_layer_norm=False)
        self.bilstm = nn.LSTM(1024, hidden_dim // 2,
                              bidirectional=True, batch_first=True)
        # self.dropout_feats = nn.Dropout(0.1)

        self.hidden2tag = nn.Linear(hidden_dim, self.tag_num)

        # T[i, j] = score of transitioning from j to i
        self.T = nn.Parameter(torch.randn(self.tag_num, self.tag_num), requires_grad=True)

        self.T.data[tag2idx[START_TAG], :] = -10000.0
        self.T.data[:, tag2idx[STOP_TAG]] = -10000.0

        self.hidden = self.init_hidden()

    def init_hidden(self):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden_a = torch.randn(1, batch_size, self.hidden_dim // 2, device=device)
        hidden_b = torch.randn(1, batch_size, self.hidden_dim // 2, device=device)

        hidden_a = autograd.Variable(hidden_a)
        hidden_b = autograd.Variable(hidden_b)

        return hidden_a, hidden_b

    def forward_alg(self, feats, mask):
        '''
        :param feats: b, l, tag
        :param mask: b, l
        :return:
        '''
        bs, seq_len, _ = feats.shape
        # print(bs, seq_len)
        mask = mask.transpose(1, 0).contiguous()
        ins_num = seq_len * bs
        feats = feats.transpose(0, 1).contiguous().view(ins_num, 1, self.tag_num).expand(ins_num, self.tag_num, self.tag_num)
        scores = feats + self.T.view(1, self.tag_num, self.tag_num).expand(ins_num, self.tag_num, self.tag_num)
        scores = scores.view(seq_len, bs, self.tag_num, self.tag_num)
        seq_iter = enumerate(scores)
        _, inivalues = next(seq_iter)
        partition = inivalues[:, self.tag2idx[START_TAG], :].clone().view(bs, self.tag_num, 1)

        for idx, cur_values in seq_iter:
            cur_values = cur_values + partition.contiguous().view(bs, self.tag_num, 1).expand(-1, -1, self.tag_num)
            cur_partition = log_sum_exp(cur_values, self.tag_num)
            mask_idx = mask[idx, :].view(bs, 1).expand(-1, self.tag_num)
            masked_cur_partition = cur_partition.masked_select(mask_idx)
            mask_idx = mask_idx.contiguous().view(bs, self.tag_num, 1)
            partition.masked_scatter_(mask_idx, masked_cur_partition)

        cur_values = self.T.view(1, self.tag_num, self.tag_num).expand(bs, -1, -1) + partition.contiguous().view(bs, self.tag_num, 1).expand(-1, -1, self.tag_num)
        cur_partition = log_sum_exp(cur_values, self.tag_num)
        final_partition = cur_partition[:, self.tag2idx[STOP_TAG]]
        return final_partition.sum(), scores

    def get_feature(self, seq, lens):
        bs = len(seq)
        self.hidden = self.init_hidden()
        ids = batch_to_ids(seq).to(device)
        elmo_out = self.elmo(ids)
        embedded = elmo_out['elmo_representations'][0]
        mask = elmo_out['mask']
        lstm_input = pack_padded_sequence(embedded, lens, batch_first=True, enforce_sorted=False)
        output, _ = self.bilstm(lstm_input)
        output, _ = pad_packed_sequence(output, batch_first=True)
        output = output.contiguous().view(-1, output.shape[2])
        # output = self.dropout_feats(output)
        output = self.hidden2tag(output)
        return output.view(bs, -1, self.tag_num), mask.bool()

    def score_sentence(self, scores, mask, tags):
        seq_len, bs, tag_num, _ = scores.shape
        tags = tags.view(bs, seq_len)
        new_tags = torch.empty(bs, seq_len, device=device, requires_grad=True).long()
        for idx in range(seq_len):
            if idx == 0:
                new_tags[:, 0] = (tag_num - 2) * tag_num + tags[:, 0]
            else:
                new_tags[:, idx] = tags[:, idx - 1] * tag_num + tags[:, idx]
        end_transition = self.T[:, self.tag2idx[STOP_TAG]].contiguous().view(1, tag_num).expand(bs, -1)
        length_mask = torch.sum(mask, dim=1).view(bs, 1).long()
        end_ids = torch.gather(tags, 1, length_mask - 1)
        end_score = torch.gather(end_transition, 1, end_ids)
        new_tags = new_tags.transpose(0, 1).contiguous().view(seq_len, bs, 1)
        tg_energy = torch.gather(scores.view(seq_len, bs, -1), 2, new_tags).view(seq_len, bs)
        tg_energy = tg_energy.masked_select(mask.transpose(0, 1))

        return tg_energy.sum() + end_score.sum()

    def viterbi_decode(self, feats, mask):
        bs, seq_len, tag_num = feats.shape
        length_mask = torch.sum(mask.long(), dim=1).view(bs, 1).long()
        mask = mask.transpose(0, 1).contiguous()
        ins_num = bs * seq_len
        feats = feats.transpose(0, 1).contiguous().view(ins_num, 1, tag_num).expand(-1, tag_num, -1)
        scores = feats + self.T.view(1, tag_num, tag_num).expand(ins_num, -1, -1)
        scores = scores.view(seq_len, bs, tag_num, tag_num)

        seq_iter = enumerate(scores)

        backpointers = []
        partition_hist = []
        mask = (1 - mask.long()).byte()
        _, inivalues = next(seq_iter)
        partition = inivalues[:, self.tag2idx[START_TAG], :].clone().view(bs, tag_num)
        partition_hist.append(partition)
        for idx, cur_values in seq_iter:
            cur_values = cur_values + partition.contiguous().view(bs, tag_num, 1).expand(-1, -1, tag_num)
            partition, cur_bp = torch.max(cur_values, dim=1)
            partition_hist.append(partition)
            cur_bp.masked_fill_(mask[idx].view(bs, 1).expand(-1, tag_num).bool(), 0)
            backpointers.append(cur_bp)
        partition_hist = torch.cat(partition_hist, 0).view(seq_len, bs, -1).transpose(0, 1).contiguous()
        last_position = length_mask.view(bs, 1, 1).expand(bs, 1, tag_num) - 1
        last_partition = torch.gather(partition_hist, 1, last_position).view(bs, tag_num, 1)
        last_values = last_partition.expand(-1, -1, tag_num) + self.T.view(1, tag_num, tag_num).expand(bs, -1, -1)
        _, last_bp = torch.max(last_values, dim=1)
        pad_zero = torch.zeros(bs, tag_num, device=device, requires_grad=True).long()
        backpointers.append(pad_zero)
        backpointers = torch.cat(backpointers).view(seq_len, bs, tag_num)

        pointer = last_bp[:, self.tag2idx[STOP_TAG]]
        insert_last = pointer.contiguous().view(bs, 1, 1).expand(-1, -1, tag_num)
        backpointers = backpointers.transpose(0, 1).contiguous()
        backpointers.scatter_(1, last_position, insert_last)
        backpointers = backpointers.transpose(0, 1).contiguous()
        decode_idx = torch.empty(seq_len, bs, device=device, requires_grad=True).long()
        decode_idx[-1] = pointer.detach()
        for idx in range(len(backpointers) - 2, -1, -1):
            pointer = torch.gather(backpointers[idx], 1, pointer.contiguous().view(bs, 1))
            decode_idx[idx] = pointer.detach().view(bs)
        return decode_idx.transpose(0, 1)

    def neg_log_likelihood(self, sentence, tags, lens):
        feats, mask = self.get_feature(sentence, lens)
        forward_score, scores = self.forward_alg(feats, mask)
        gold_score = self.score_sentence(scores, mask, tags)
        return forward_score - gold_score

    def forward(self, sentence, lens):
        feats, mask = self.get_feature(sentence, lens)
        return self.viterbi_decode(feats, mask), mask

    def save_model(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        file = os.path.join(path, 'model.pt')
        torch.save(self.state_dict(), file)

    def load_model(self, file):
        self.load_state_dict(torch.load(file))
