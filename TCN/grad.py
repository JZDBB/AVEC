from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.externals import joblib
import numpy as np
from model import TCN
import torch
from torch import nn
import torch.optim as optim


def adjust_learning_rate(optimizer, gamma, step_index, iteration):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if iteration < 10000:
        lr = 1e-8 + (1e-6-1e-8) * iteration / 10000
    else:
        lr = 1e-6 * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

class mDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.model = joblib.load('word.pkl')
        self.csv_file = pd.read_csv(csv_file)
        id = self.csv_file.iloc[:, 0]
        label_ = self.csv_file.iloc[:, 1]
        phq_ = self.csv_file.iloc[:, 2]
        time_per_word = self.csv_file.iloc[:, 3]
        feat_ = self.csv_file.iloc[:, 4:].as_matrix()
        feat = {}
        label = {}
        phq = {}
        for i in range(feat_.shape[0]):
            curr = np.array([self.model[f] for f in feat_[i, 0].split(' ')])
            curr = np.transpose(curr, (1, 0))
            try:
                feat[id[i]].append(curr)
            except:
                feat[id[i]] = [curr]
            label[id[i]] = label_[i]
            phq[id[i]] = phq_[i]
        self.feat = feat
        self.label = label
        self.phq = phq
        self.keys = list(label.keys())
        self.num_people = len(self.keys)

    def __len__(self):
        return 99999999

    def __getitem__(self, idx):
        # np.random.seed(idx)
        id = self.keys[np.random.randint(0, self.num_people, 1)[0]]
        pool = self.feat[id]

        ret = []
        for i in range(10):
            dice = np.random.randint(0, len(pool), 1)[0]
            if len(pool[dice])>15:
                ret.append(pool[dice])
        return ret, self.label[id], self.phq[id]

def main():
    a = mDataset('../gensim/WORD.csv', '../gensim')
    b = iter(DataLoader(a, batch_size=1, shuffle=True, num_workers=1))

    model = TCN(300, 300, [512, 1024, 512, 512], 2, dropout=0.2)
    model.cuda()

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-8, weight_decay=1e-4)

    writer = model.writer

    i = 0
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    while True:
        optimizer.zero_grad()
        feat, label, _ = next(b)
        feat = [f.cuda() for f in feat]
        label = label.cuda()
        logits = model([feat, i])
        loss = criterion(logits, label)
        loss.backward()
        nn.utils.clip_grad_norm(filter(lambda p: p.requires_grad, model.parameters()), max_norm=0.2)
        optimizer.step()
        torch.save(model.state_dict(), '')

        if i % 1 == 0:
            prediction = torch.argmax(logits, dim=-1)
            writer.add_histogram("prediction", prediction.cpu().data.numpy(), i)
            writer.add_scalar('loss', loss.item(), i)
        # adjust_learning_rate(optimizer, 0.997, 100, i)
        i += 1

if __name__ == '__main__':
    main()
