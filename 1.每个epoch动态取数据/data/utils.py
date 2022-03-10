import logging
import os
import time


def setuplogger(dir_label, embedding_dim, batch_size, lr, drop_rate, weight_decay, mode):
    log_code = None
    if 'train' in mode or 'load' in mode:
        log_code = 'train'
    if 'test' in mode:
        log_code = 'test'

    formatter = logging.Formatter("[%(levelname)s %(asctime)s] %(message)s")

    Log_file = logging.getLogger('Log_file')
    Log_screen = logging.getLogger('Log_screen')

    log_path = os.path.join('./logs_' + dir_label + '_' + log_code)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file_name = os.path.join(log_path, f'log_bs_{batch_size}'
                                           f'_ed_{embedding_dim}_lr_{lr}_dp{drop_rate}_wd_{weight_decay}-'
                                 + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.log')
    Log_file.setLevel(logging.INFO)
    Log_screen.setLevel(logging.INFO)

    th = logging.FileHandler(filename=log_file_name, encoding='utf-8')
    th.setLevel(logging.INFO)
    th.setFormatter(formatter)
    Log_file.addHandler(th)

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)

    Log_screen.addHandler(handler)
    Log_file.addHandler(handler)
    return Log_file, Log_screen