# -*-coding:utf-8-*-
import numpy as np
import random
from multiprocessing import Pool

def random_select(seed):
    print(seed)
    # local_state = np.random.RandomState(seed)
    np.random.seed(seed)
    return np.random.randint(0,200)

# 开启8个进程
pool = Pool(processes=8)
# 运行20次random_select函数
print(np.array(pool.map(random_select, range(2,20))))
