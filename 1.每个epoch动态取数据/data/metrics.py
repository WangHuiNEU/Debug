import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import BuildEvalDataset


def hit(gt_item, pred_items):
    if gt_item in pred_items:
        return 1
    return 0


def ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index + 2))
    return 0


def eval_model(model, user_history, eval_pairs, test_batch_size, args, item_num):
    eval_dataset = BuildEvalDataset(eval_pairs=eval_pairs, item_num=item_num)
    eval_dl = DataLoader(eval_dataset, batch_size=test_batch_size,
                         num_workers=args.num_workers, pin_memory=True)
    model.eval()
    with torch.no_grad():
        HR, NDCG = [], []
        all_item = torch.LongTensor(np.array(list(range(0, item_num)))).cuda()
        for data in eval_dl:
            user_ids, targets = data
            user_ids, targets = user_ids.cuda(), targets.cuda()
            for user_id, target in zip(user_ids, targets):
                user_id = user_id[0].item()
                user_list = torch.LongTensor(np.array([user_id] * item_num)).cuda()
                history = torch.LongTensor(np.array(user_history[user_id])).cuda()
                prediction_i, _ = model(user_list, all_item, all_item)
                prediction_i[history] = 0
                _, indices = torch.topk(prediction_i, 10)
                recommends = torch.take(all_item, indices).cpu().numpy().tolist()
                gt_item = target[0].item()
                HR.append(hit(gt_item, recommends))
                NDCG.append(ndcg(gt_item, recommends))
    return np.mean(HR), np.mean(NDCG)
