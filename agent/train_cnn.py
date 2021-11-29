import argparse

import torch
import torch.nn as nn
import torch.functional as F
from dl_bot import ConvNet


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--nepochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--use-gpu", action="store_true", default=False)
    parser.add_argument("--save-model", type=str, default="")

    args = parser.parse_args()

    use_gpu = args.use_gpu and torch.cuda.is_available()

    if  use_gpu:
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")

    net = ConvNet()

    if use_gpu:
        net.to(device)

    opts = {
        "sgd": torch.optim.SGD,
        "adagrad": torch.optim.Adagrad,
        "RMSprop": torch.optim.RMSprop,
        "Adadelta" : torch.optim.Adadelta,
        "Adam" :torch.optim.SparseAdam,
        "AdaMax" : torch.optim.Adamax,
        "ASGD" : torch.optim.ASGD
    }
    optimizer = opts[args.optimizer](params=net.parameters(), lr= args.lr)
    criterion = nn.CrossEntropyLoss()
    Losses = []
    acc = []