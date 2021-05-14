import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import re
from torch.utils.data import Dataset, DataLoader
import pickle as pk
import torch.nn.functional as F
import glob

max_word_dim = 386
hidden_dim = 512

class DaconDataset(Dataset):
    def __init__(self, data_path, words=None):
        super(DaconDataset, self).__init__()
        self.train = pd.read_csv(data_path)
        self.train_x = np.array(self.train['full_log'])
        self.id = np.array(self.train['id'])
        self.vocab_table = words
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
        return sentence, self.id[index]

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


class StackedGRU(torch.nn.Module):
    def __init__(self, n_tags, len_word):
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
        self.embedding = nn.Embedding (len_word, hidden_dim).cuda ()
        # self.ae = AE()
    def get_embedding(self):
        return self.embedding

    def forward(self, x):
        x = self.embedding(x)
        outs, hidden = self.rnn(x)
        hidden = torch.mean(hidden.transpose(0, 1), dim=1)
        out = self.fc(hidden)
        # ae_out = self.ae(hidden)
        return hidden, out


def main():
    vocab_list = []
    vocab_glob = sorted(glob.glob("*.pkl"))

    for vg in vocab_glob:
        with open(vg, 'rb') as f:
            vocab_list.append(pk.load(f))

    model_list = []
    model_glob = sorted(glob.glob("*7*.pth"))

    for idx, mg in enumerate(model_glob):
        model = StackedGRU(n_tags=7, len_word=len(vocab_list[idx])).cuda()
        model = nn.DataParallel(model, device_ids=[0])
        model.load_state_dict(torch.load(mg))
        model_list.append(model)
    model_list = nn.ModuleList(model_list)
    model_list.eval()

    dataset1 = DaconDataset('./data/test.csv', vocab_list[0])
    dataset2 = DaconDataset('./data/test.csv', vocab_list[1])
    dataset3 = DaconDataset('./data/test.csv', vocab_list[2])
    dataset4 = DaconDataset('./data/test.csv', vocab_list[3])
    dataset5 = DaconDataset('./data/test.csv', vocab_list[4])

    dataloader1 = DataLoader(dataset1, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)
    dataloader2 = DataLoader(dataset2, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)
    dataloader3 = DataLoader(dataset3, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)
    dataloader4 = DataLoader(dataset4, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)
    dataloader5 = DataLoader(dataset5, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)

    gru_threshold = 0.88
    submission_dict = {}
    with torch.no_grad():
        for idx, (test1, test2, test3, test4, test5) in enumerate(zip(dataloader1, dataloader2, dataloader3, dataloader4, dataloader5)):
            data1, id1 = test1
            data2, id2 = test2
            data3, id3 = test3
            data4, id4 = test4
            data5, id5 = test5
            data1.cuda(); data2.cuda(); data3.cuda(); data4.cuda(); data5.cuda()

            _, logit1 = model_list[0](data1)
            _, logit2 = model_list[1](data2)
            _, logit3 = model_list[2](data3)
            _, logit4 = model_list[3](data4)
            _, logit5 = model_list[4](data5)
            logit_list = torch.stack([logit1, logit2, logit3, logit4, logit5])
            logit_soft = F.softmax(logit_list, dim=-1)
            final_logit = torch.mean(logit_soft, dim=0)
            _, final_prediction = torch.max(final_logit, 1)

            transposed_logit_list = logit_soft.permute((1, 0, 2))

            for prediction, transposed_logit, id in zip(final_prediction, transposed_logit_list, id1):
                confidence_list = []
                for each_logit in transposed_logit:
                    if each_logit[prediction] == torch.max(each_logit):
                        confidence_list.append(torch.max(each_logit))

                final_confidence = sum(confidence_list) / len(confidence_list)
                if final_confidence >= gru_threshold:
                    submission_dict[id.item()] = int(prediction)
                else:
                    submission_dict[id.item()] = 7


        submission_dict = sorted (submission_dict.items ())
        submission_value = [l[1] for l in submission_dict]
        submission = pd.read_csv ('./data/sample_submission.csv')
        submission['level'] = submission_value
        print (submission.head ())
        submission.to_csv ('fifth_submission.csv', index=False)
if __name__ == "__main__":
    main()