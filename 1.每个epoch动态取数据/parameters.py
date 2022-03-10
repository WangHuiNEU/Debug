from data.utils import *
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # ============== data_dir ==============
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--item_tower", type=str, default="id")

    # ============== train parameters ==============
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--drop_rate", type=float)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)

    # ============== model parameters ==============
    parser.add_argument("--embedding_dim", type=int)

    # ============== switch and logging setting ==============
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--label_screen", type=str)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
