import os


mode = 'train'
item_tower = 'id'

epoch = 40


weight_decay_list = [1e-2]
drop_rate_list = [0.2]
batch_size_list = [1]
lr_list = [1e-3]
embedding_dim_list = [64]

for batch_size in batch_size_list:
    for embedding_dim in embedding_dim_list:
        for lr in lr_list:
            for drop_rate in drop_rate_list:
                for weight_decay in weight_decay_list:
                    label_screen = '{}_bs{}_ed{}_lr{}_dp{}_wd{}'.format(item_tower,
                                                                        batch_size, embedding_dim, lr, drop_rate,
                                                                        weight_decay)
                    run_py = "python run.py --mode {} --epoch {} --item_tower {}\
                             --batch_size {} --lr {} --weight_decay {}\
                             --embedding_dim {}  --drop_rate {} \
                             --label_screen {}".format(
                        mode, epoch, item_tower,
                        batch_size, lr, weight_decay,
                        embedding_dim, drop_rate, label_screen)
                    os.system(run_py)

