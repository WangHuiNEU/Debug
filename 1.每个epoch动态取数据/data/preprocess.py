import numpy as np
import pandas as pd
import scipy.sparse as sp


def read_behaviors(train_data, test_data):
    train_data = pd.read_csv(
        train_data,
        sep='\t', header=None, names=['user', 'item'],
        usecols=[0, 1], dtype={0: np.int32, 1: np.int32})

    user_num = train_data['user'].max() + 1
    item_num = train_data['item'].max() + 1

    train_data = train_data.values.tolist()

    # load ratings as a dok matrix
    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)

    user_his = {}
    train_pairs = []
    for x in train_data:
        user_id = x[0]
        item_id = x[1]
        train_mat[user_id, item_id] = 1.0
        train_pairs.append((user_id, item_id))
        if user_id not in user_his:
            user_his[user_id] = [item_id]
        else:
            user_his[user_id].append(item_id)

    test_pairs = []
    with open(test_data, 'r') as fd:
        line = fd.readline()
        while line != None and line != '':
            arr = line.split('\t')
            test_pairs.append((int(arr[0]), int(arr[1])))
            line = fd.readline()
    return user_num, item_num, user_his, train_mat, np.array(train_pairs),  np.array(test_pairs), train_data