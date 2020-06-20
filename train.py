from model import *
from param import *
from helper import *
import time
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from seqeval.metrics import accuracy_score, f1_score


get_lr = lambda epoch: 1 / (1 + 0.05 * epoch)


class trainer:
    def __init__(self, train_data, valid_data, test_data, vocab, char_vocab_size, embedding_dim,
                 char_embedding_dim, hidden_dim, window_size, filter_num):
        vocab_size = len(vocab)
        self.model = BiLSTM_CNN_CRF(vocab_size, char_vocab_size, embedding_dim,
                                    char_embedding_dim, hidden_dim, window_size, filter_num).to(device)
        self.loss_func = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.015, momentum=0.9)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, get_lr)
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data

        self.timemark = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
        self.result_path = os.path.join('./result', self.timemark)

    def train(self, epochs=20):
        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            tqdm_batch = tqdm(build_dataloader(self.train_data, is_eval=False))
            total_loss = 0
            eg_num = 0
            for batch in tqdm_batch:
                self.optimizer.zero_grad()
                seq = batch["seq"]
                tag = batch["tag"]
                lens = batch["leng"]
                loss = self.model.neg_log_likelihood(seq, tag, lens)
                # logit = self.model.get_feature(seq, char_index, lens)
                # bs, seq_len, _ = logit.size()
                # print(bs, seq_len)
                # loss = self.loss_func(logit.view(bs * seq_len, -1), tag.view(-1))
                loss.backward()
                clip_grad_norm_(self.model.parameters(), max_norm=clipping)
                self.optimizer.step()
                self.scheduler.step(epoch)
                total_loss += loss.item()
                eg_num += len(seq)
                tqdm_batch.set_description("epoch: {}, avg_loss: {:.6f}".format(epoch, total_loss / eg_num))
            self.eval(epoch)

    def eval(self, epoch):
        self.model.eval()
        train = self.batch_eval(self.train_data)
        valid = self.batch_eval(self.valid_data)
        test = self.batch_eval(self.test_data)
        path = self.result_path

        result = "epoch{}_train_{:.4f}_{:.4f}_valid_{:.4f}_{:.4f}_test_{:.4f}_{:.4f}".format(
            epoch, train[0], train[1], valid[0], valid[1], test[0], test[1]
        )
        print(result)
        if not os.path.exists('./result'):
            os.mkdir('./result')
        if not os.path.exists(self.result_path):
            os.mkdir(self.result_path)
        open(os.path.join(path, result), 'w')

    def batch_eval(self, data):
        loader = build_dataloader(data, is_eval=True)
        with torch.no_grad():
            golds = []
            predicts = []
            tqdm_batch = tqdm(loader)
            for batch in tqdm_batch:
                seq = batch["seq"]
                tag = batch["tag"]
                lens = batch["leng"]
                predict, mask = self.model(seq, lens)
                predict = decode(predict, idx2tag, mask)
                golds.extend(tag)
                predicts.extend(predict)
            print(golds[0])
            print(predicts[0])
            acc = accuracy_score(golds, predicts)
            f1 = f1_score(golds, predicts)
            return acc, f1
