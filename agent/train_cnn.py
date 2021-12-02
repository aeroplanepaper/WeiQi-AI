import argparse
import os
import random

import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader
from agent.dl_bot import ConvNet2D
from agent.dataProcessor import TrainDataSet, TestDataset


def test_model(model, testDataLoader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for idx, data in enumerate(testDataLoader):
            test_data = data[0]
            test_data = test_data.float()
            test_label = data[1]

            if total > 10000:
                break
            test_data = test_data.to(device)
            test_label = test_label.to(device)
            outputs = model(test_data)
            _, outputs = outputs.max(dim=1)
            outputs = outputs + 1
            correct += (outputs == test_label).sum().item()
            total += test_label.size(0)
        print('Test Accuracy of the model on 10000 tests: {} %'.format(100 * correct / total))



if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--nepochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--use-gpu", action="store_true", default=True)
    parser.add_argument("--save-model", type=str, default="")
    parser.add_argument("--test-freq", type=int, default=2048)
    parser.add_argument("--print-freq", type=int, default=100)
    args = parser.parse_args()

    use_gpu = args.use_gpu and torch.cuda.is_available()
    # print(use_gpu)
    if  use_gpu:
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")

    model = ConvNet2D()

    if use_gpu:
        model.to(device)

    opts = {
        "sgd": torch.optim.SGD,
        "adagrad": torch.optim.Adagrad,
        "RMSprop": torch.optim.RMSprop,
        "Adadelta" : torch.optim.Adadelta,
        "Adam" :torch.optim.Adam,
        "AdaMax" : torch.optim.Adamax,
        "ASGD" : torch.optim.ASGD
    }
    optimizer = opts[args.optimizer](params=model.parameters(), lr= args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    criterion = nn.CrossEntropyLoss()
    Losses = []
    Acc = []

    path = os.path.abspath("./data")
    data_path = []

    for dir_path, dir_names, file_names in os.walk(path):
        for file_name in file_names:
            file_path = dir_path + "\\" + file_name
            data_path.append(file_path)
    len_data = len(data_path)
    random.shuffle(data_path)

    train_data_path = data_path[0: int(len_data*0.7)]
    test_data_path = data_path[int(len_data * 0.7) : len_data]

    trainDataSet = TrainDataSet(data_path=train_data_path, batch_size=args.batch_size)
    testDataSet = TestDataset(data_path=test_data_path, batch_size=args.batch_size)
    trainDataLoader = DataLoader(trainDataSet, batch_size=args.batch_size)
    testDataLoader = DataLoader(testDataSet, batch_size=args.batch_size)

    iteration = 0
    num_of_data = 0
    correct = 0
    for epoch in range(args.nepochs):
        for idx, data in enumerate(trainDataLoader):
            # print(len(trainDataLoader))
            train_data = data[0]
            train_data = train_data.float()
            train_label = data[1]
            optimizer.zero_grad()
            # print(train_data, train_label)
            train_data = train_data.to(device)
            train_label = train_label.to(device)
            if iteration % args.test_freq == 0 and iteration != 0:
                test_model(model, testDataLoader)


            num_of_data += args.batch_size
            outputs = model(train_data)
            # print(outputs)
            _, temp_out = outputs.max(dim=1)
            temp_out = temp_out + 1
            # outputs = outputs.float()
            # print(outputs.dtype)
            train_label = train_label.long()
            # print(train_label, temp_out)
            # print(outputs, train_label)
            loss = criterion(outputs, train_label)
            correct += (temp_out == train_label).sum().item()
            acc = correct / num_of_data

            Acc.append(acc)
            Losses.append(loss.item())
            loss.backward()
            optimizer.step()
            if iteration % args.print_freq == 0 and iteration != 0:
                print(iteration, ' ', loss.item(),' ', acc)
                # print("At iteration {:} , loss: {:%4f} , ".format(iteration, loss.item()))
            iteration += 1