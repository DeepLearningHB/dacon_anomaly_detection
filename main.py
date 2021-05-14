import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from GC.RAdam import RAdam2, Lookahead
import re
from utils.ca_warnup import *
from utils.utils import AverageMeter
from sklearn.metrics import confusion_matrix, f1_score
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
word_dim = -1
import pickle as pk
max_word_dim = 386
hidden_dim = 512

def word_table(csv_file):
    train = pd.read_csv (csv_file)
    logs = train['full_log'].str.replace (r'[0-9]', '')
    logs = logs.str.replace ('[^a-zA-Z]', ' ')
    logs = [list (set (a.lower ().split (" ")))[1:] for a in logs]
    words = []
    for log in logs:
        words.extend (log)
    vocabs = list (set (words))
    vocab = {tkn: i + 2 for i, tkn in enumerate (vocabs)}  # 단어 집합의 각 단어에 고유한 정수 맵핑.
    vocab['<unk>'] = 0
    vocab['<pad>'] = 1
    word_dim = len (vocab)
    return word_dim, vocab

def make_context_vector(context, vocab_):
    indices = []
    for word in context:
        try:
            indices.append (vocab_[word])
        except KeyError:  # 단어 집합에 없는 단어일 경우 <unk>로 대체된다.
            indices.append (vocab_['<unk>'])
    pad_length = max_word_dim - len(indices)
    for i in range(pad_length):
        indices.append(vocab_['<pad>'])
    return torch.tensor(indices, dtype=torch.long)

class DaconDataset(Dataset):
    def __init__(self, data_path, is_train=True, words=None):
        super(DaconDataset, self).__init__()
        global word_dim
        self.train = pd.read_csv(data_path)
        # self.train = self.train.loc[self.train['level'] < 6]
        self.train_x = np.array(self.train['full_log'])
        self.train_y = np.array(self.train['level'])
        self.is_train = is_train
        if self.is_train:
            self.word_dim, self.vocab_table = word_table (data_path)
            word_dim = self.word_dim
        else:
            self.vocab_table = words
        if self.is_train:
            self.value_dict = {}
            for idx, label in enumerate(self.train_y):
                if label not in self.value_dict.keys():
                    self.value_dict[label] = [idx]
                else:
                    self.value_dict[label].append(idx)
        self.max_word_counts = max_word_dim

    def __len__(self):
        return len(self.train)

    def sentence_to_indices(self, sentence):
        sentence = re.sub (r'[0-9]', '', sentence)
        sentence = re.sub ('[^a-zA-Z]', ' ', sentence)
        sentence = str (sentence).lower ()
        sentence = ' '.join (sentence.split ()[:self.max_word_counts]).split ()
        sentence_vector = make_context_vector (sentence, self.vocab_table)
        return sentence_vector



    def __getitem__(self, index):

        sentence = self.sentence_to_indices(self.train_x[index])
        sentence_y = self.train_y[index]

        if self.is_train:
            #positive sample
            p_index = np.random.randint(0, len(self.value_dict[sentence_y]))
            sentence_p = self.sentence_to_indices(self.train_x[self.value_dict[sentence_y][p_index]])
            sentence_py = self.train_y[index]

            #negative sample
            n_list = [i for i in range(7) if i != sentence_y]
            select_negative = np.random.randint(0, len(n_list))
            n_index = np.random.randint(0, len(self.value_dict[n_list[select_negative]]))
            sentence_n = self.sentence_to_indices(self.train_x[self.value_dict[n_list[select_negative]][n_index]])
            sentence_ny = self.train_y[self.value_dict[n_list[select_negative]][n_index]]

            return sentence, sentence_y, sentence_p, sentence_py, sentence_n, sentence_ny
        return sentence, sentence_y


def custom_collate_fn(batch):
    return [[(data[0][0], data[0][1]), (data[1][0], data[1][1]), (data[2][0], data[2][1])] for data in batch]


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.ae = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear (hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d (hidden_dim // 4),
            nn.ReLU (),
            nn.Linear (hidden_dim // 4, hidden_dim // 2),
            nn.BatchNorm1d (hidden_dim // 2),
            nn.ReLU (),
            nn.Linear (hidden_dim // 2, hidden_dim)
        )


    def forward(self, input_feat):
        return self.ae(input_feat)

class StackedGRU(torch.nn.Module):
    def __init__(self, n_tags):
        super().__init__()
        self.rnn = torch.nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=False,
            dropout=0.5,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, n_tags)
        self.embedding = nn.Embedding (word_dim, hidden_dim).cuda ()
        # self.ae = AE()

    def forward(self, x):
        x = self.embedding(x)
        outs, hidden = self.rnn(x)
        hidden = torch.mean(hidden.transpose(0, 1), dim=1)
        out = self.fc(hidden)
        # ae_out = self.ae(hidden)
        return hidden, out


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad ():
        maxk = max (topk)
        batch_size = target.size (0)

        _, pred = output.topk (maxk, 1, True, True)
        pred = pred.t ()
        correct = pred.eq (target.view (1, -1).expand_as (pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape (-1).float ().sum (0, keepdim=True)
            res.append (correct_k.mul_ (100.0 / batch_size))
        return res


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, logits, labels):
        y_pred = F.softmax(logits, dim=-1)
        labels = F.one_hot(labels, num_classes=7).type(torch.float32)
        loss = - labels * ((1 - y_pred) ** self.gamma) * torch.log(y_pred)
        loss = torch.mean(torch.sum(loss, dim=1))
        return loss

if __name__ == "__main__":
    train_dataset = DaconDataset('./data/k_fold/train_1.csv')
    vocabs = train_dataset.vocab_table
    with open('vocab_table.pkl', 'wb') as f:
        pk.dump(vocabs, f)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, pin_memory=True, num_workers=8)

    valid_dataset = DaconDataset('./data/k_fold/valid_1.csv', is_train=False, words=vocabs)
    valid_loader = DataLoader(valid_dataset, batch_size=1024, shuffle=True, pin_memory=True, num_workers=8)

    model = StackedGRU(n_tags=7).cuda()
    model = nn.DataParallel(model, device_ids=[0,1,2,3])
    optimizer = Lookahead (RAdam2 (filter (lambda p: p.requires_grad, model.parameters()), lr=0.001),
                           alpha=0.5,
                           k=5)
    # optimizer = Lookahead (torch.optim.SGD (model.parameters (), lr=0.001, momentum=0.9),
    #                        alpha=0.5,
    #                        k=5)
    scheduler = CosineAnnealingWarmupRestarts (optimizer, first_cycle_steps=20, cycle_mult=1.0, max_lr=0.001,
                                               min_lr=0.00005, warmup_steps=5, gamma=0.8)


    criterion_TP = nn.TripletMarginLoss(margin=0.7)
    criterion_CE = FocalLoss(gamma=2)
    epoch = 70
    best_acc = -1
    for ep in range(epoch):
        avg_triplet = AverageMeter()
        avg_ce = AverageMeter()
        avg_acc = AverageMeter()
        model.train()
        for data in train_loader:
            origin_x, origin_y = data[0].cuda(), data[1].cuda()
            positive_x, positive_y = data[2].cuda(), data[3].cuda()
            negative_x, negative_y = data[4].cuda(), data[5].cuda()

            origin_x_feat, origin_y_hat = model(origin_x)
            positive_x_feat, positive_y_hat = model(positive_x)
            negative_x_feat, negative_y_hat = model(negative_x)
            # loss_tp = torch.FloatTensor([0.]).cuda()
            loss_tp = criterion_TP (origin_x_feat, positive_x_feat, negative_x_feat)
            loss_ce = (criterion_CE (origin_y_hat, origin_y) + criterion_CE (positive_y_hat, origin_y) + criterion_CE (
                negative_y_hat, negative_y)) / 3

            acc_1, acc_3 = accuracy(origin_y_hat, origin_y, topk=(1, 3))
            avg_acc.update(acc_1.item(), data[0].shape[0])
            avg_triplet.update(loss_tp.item(), data[0].shape[0])
            avg_ce.update(loss_ce.item(), data[0].shape[0])

            loss = loss_tp + loss_ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        y_hat_list = []
        y_list = []
        model.eval()
        with torch.no_grad():
            avg_valid_acc1 = AverageMeter()
            avg_valid_acc3 = AverageMeter()
            avg_val_ce = AverageMeter()
            for data in valid_loader:
                valid_x, valid_y = data[0].cuda(), data[1].cuda()
                y_list.extend(valid_y.cpu().detach().numpy().tolist())
                valid_x_feat, valid_y_hat = model(valid_x)
                _, y_hat = torch.max(valid_y_hat, 1)
                y_hat_list.extend(y_hat.cpu().detach().numpy().tolist())
                acc_1, acc_3 = accuracy(valid_y_hat, valid_y, topk=(1, 3))

                avg_valid_acc1.update(acc_1[0], data[0].shape[0])
                avg_valid_acc3.update(acc_3[0], data[0].shape[0])
                loss_ = criterion_CE(valid_y_hat, valid_y)
                avg_val_ce.update(loss_.item(), data[0].shape[0])

            print(confusion_matrix(y_list, y_hat_list))

        macro_avg = f1_score(y_list, y_hat_list, average='macro')
        print("[Train | Epoch:%02d] ce: %.4f tp: %.4f Acc.: %.4f \n[Valid] ce: %.4f  Acc.@1:%.4f Acc.@3: %.4f MacroAvg: %.4f" % (
            ep, avg_ce.avg, avg_triplet.avg, avg_acc.avg, avg_val_ce.avg, avg_valid_acc1.avg, avg_valid_acc3.avg, macro_avg
        ))
        if best_acc < macro_avg:
            print("Accuracy has been improved. %.4f --> %.4f" % (best_acc, macro_avg))
            torch.save(model.state_dict(), "best_gru_7_cls_fold1.pth")
            best_acc = macro_avg




