import torch
import torch.multiprocessing as mp
mp.set_start_method('spawn')
import torch.optim as optim

import argparse
import re
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np

from parameters import parse_args
from model import Model
from data import read_behaviors, BuildTrainDataset, eval_model, BuildTrainDataset, BuildTrainDataset_cc
from data.utils import *
import torch.backends.cudnn as cudnn
import random



os.environ["TOKENIZERS_PARALLELISM"] = "false"
GLOBAL_SEED = 1
def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

def worker_init_fn(worker_id):
  global GLOBAL_WORKER_ID
  GLOBAL_WORKER_ID = worker_id
  set_seed(GLOBAL_SEED + worker_id)

def train(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    Log_file.info('read behaviors...')
    # train_data = 'ml1m/ml-1m.train.rating'
    # test_data = 'ml1m/ml-1m.test.rating'
    train_data = 'ml100k/ml100k_train.csv'
    test_data = 'ml100k/ml100k_test.csv'
    user_num, item_num, user_his, train_mat,  train_pairs, test_pairs, train_data = read_behaviors(train_data, test_data)
    Log_file.info('user_num {},  item_num {}, user_his {}, train_mat {},  train_pairs {}, test_pairs {}, train_data {}'
                  .format(user_num, item_num, len(user_his), len(train_mat),  len(train_pairs), len(test_pairs), len(train_data)))
    Log_file.info('build dataset...')
    train_dataset = BuildTrainDataset_cc(train_mat=train_mat, train_pairs=train_pairs, item_num=item_num,
                                      train_data=train_data, user_his=user_his)

    Log_file.info('build dataloader...')
    train_dl = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=16,
                          pin_memory=True, shuffle=False) #,worker_init_fn = worker_init_fn)

    Log_file.info('build model...')
    model = Model(args, user_num, item_num).cuda()
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 0
    for ep in range(args.epoch):
        now_epoch = start_epoch + ep + 1
        Log_file.info('epoch {} start'.format(now_epoch))
        bs_loss = 0.0
        model.train()
        number = 0
        # train_dl.dataset.ng_sample()
        for data in train_dl:
            input_user, input_pos, input_neg = data
            print(input_user,"____",input_pos,"____",input_neg)
            number += 1
        print(number)

        #     input_user, input_pos, input_neg = \
        #         input_user.cuda().squeeze(), \
        #         input_pos.cuda().squeeze(), \
        #         input_neg.cuda().squeeze()
        #     prediction_i, prediction_j = model(input_user, input_pos, input_neg)
        #
        #     loss = - (prediction_i - prediction_j).sigmoid().log().sum()
        #
        #     bs_loss += loss.data.float()
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        # Log_file.info('loss: {:.5f}'.format(bs_loss.data))
        # model.eval()
        # HR, NDCG = eval_model(model, user_his, test_pairs, 1, args, item_num)
        # Log_file.info("HR: {:.3f}\tNDCG: {:.3f}".format(100 * np.mean(HR), 100 * np.mean(NDCG)))
        print("A")
        break


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    cudnn.benchmark = True
    args = parse_args()
    setup_seed(123456)
    dir_label = str(args.item_tower)
    Log_file, Log_screen = setuplogger(dir_label, args.embedding_dim, args.batch_size, args.lr, args.drop_rate,
                                       args.weight_decay, args.mode)
    Log_file.info(args)
    if 'train' in args.mode:
        train(args)
