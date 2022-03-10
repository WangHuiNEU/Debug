import torch
from torch.utils.data import Dataset
import numpy as np
import random

class BuildTrainDataset(Dataset):
    def __init__(self, train_mat, train_pairs, item_num, train_data, user_his):
        self.train_mat = train_mat
        self.train_pairs = train_pairs
        self.item_num = item_num
        self.train_data = train_data
        self.user_his = user_his

    def ng_sample(self):
        self.features_fill = []
        for x in self.train_data:
            u, i = x[0], x[1]
            j = np.random.randint(self.item_num)
            # if j in self.user_his[u]:
            while (u, j) in self.train_mat:
                j = np.random.randint(self.item_num)
            self.features_fill.append([u, i, j])

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        features = self.features_fill
        user = features[idx][0]
        item_i = features[idx][1]
        item_j = features[idx][2]
        return user, item_i, item_j

    # @staticmethod
    # def collate__fn(batch):
    #     a,b,c = tuple(zip(*batch))
    #     print(a,b,c)
    #     return 1,2,3



class BuildTrainDataset_cc(Dataset):
    def __init__(self, train_mat, train_pairs, item_num, train_data, user_his):
        self.train_mat = train_mat
        self.train_pairs = train_pairs
        self.item_num = item_num
        self.train_data = train_data
        self.user_his = user_his

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        pair = self.train_data[index]
        user_id, poss = pair[0], pair[1]
        # sam_neg = random.randint(0, self.item_num-1)
        sam_neg= np.random.randint(self.item_num)
        while (user_id, sam_neg) in self.train_mat:
            # sam_neg = random.randint(0, self.item_num-1)
            sam_neg = np.random.randint(self.item_num)
        return user_id, poss, sam_neg



class BuildEvalDataset(Dataset):
    def __init__(self, eval_pairs, item_num):
        self.eval_pairs = eval_pairs
        self.item_num = item_num

    def __len__(self):
        return len(self.eval_pairs)

    def __getitem__(self, index):
        (user_id, target) = self.eval_pairs[index]
        return torch.LongTensor([user_id]), torch.LongTensor([target])