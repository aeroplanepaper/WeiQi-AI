import os
import random
import copy
import numpy as np
from torch.utils.data import DataLoader, Dataset
from Game.Player import Color
from gosgf.sgf import Sgf_game
from Game.GameState import GameState
from Game.Board import Board,Point
from Game.Move import Move
from Game.Player import Color
from encoders.base import get_encoder_by_name
from multiprocessing import Process
from numba import jit
import threading

class TrainDataSet(Dataset):
    def __init__(self, data_path, batch_size):
        self.data_path = data_path
        self.batch = batch_size
        self.encoder = get_encoder_by_name("simple", 19)

        shape = self.encoder.shape()
        self.data = []
        self.labels = []
        self.index = 0
        self.percent = 0
        self.moves = 0
        # for idx, file_path in enumerate(data_path):
        #     if (idx % 1000 == 0):
        #         print("Data processing finished :", 100 * idx / len(self.data_path), "% .")
        #     file = open(file_path).read()
        #     sgf = Sgf_game.from_string(file)
        #     self.moves += num_total_moves(sgf)

    def __len__(self):
        return 500000

    def __getitem__(self, index):
        if self.index == len(self.data_path) -1 :
            self.index = 0
        if len(self.data) < self.batch:
            temp_data, temp_label = get_data(self.data_path, self.index)
            self.index += 1
            self.data.extend(temp_data)
            self.labels.extend(temp_label)
        sample = self.data[0]
        label = self.labels[0]
        self.data.pop(0)
        self.labels.pop(0)
        return sample, label

    # def get_data(self):
    #     length = len(self.data_path)
    #     for i in range(length):
    #         if i % 10 == 0:
    #             print(i /(length/1000), "% have done! ")
    #         file_path = self.data_path[i]
    #         # self.index += 1
    #         if not file_path.endswith(".sgf"):
    #             raise ValueError(file_path + "is not sgf file!")
    #         file = open(file_path).read()
    #         sgf = Sgf_game.from_string(file)
    #         board, first_move_done = get_handicap(sgf)
    #         if board is None:
    #             board = Board(19, 19)
    #         shape = self.encoder.shape()
    #         total_move = num_total_moves(sgf)
    #         # print(total_move)
    #         feature_shape = np.insert(shape, 0, np.asarray([total_move]))
    #         # print(feature_shape)
    #         features = np.zeros(feature_shape)
    #         labels = np.zeros(total_move)
    #         cnt = 0
    #         for item in sgf.main_sequence_iter():
    #             color, move_tuple = item.get_move()
    #             point = None
    #             if color is not None:
    #                 if move_tuple is not None:
    #                     player = Color.BLACK if color == 'b' else Color.WHITE
    #                     row, col = move_tuple
    #                     point = Point(row+1, col+1)
    #                     # move = Move(point)
    #                 else:
    #                     continue
    #                 if first_move_done and point is not None:
    #                     features[cnt] = self.encoder.encode(board, player)
    #                     labels[cnt] = self.encoder.encode_point(point)
    #                     cnt += 1
    #                 board.place_stone(player, point)
    #                 first_move_done = True
    #         self.data = np.concatenate((self.data, features), axis=0)
    #         self.labels = np.concatenate((self.labels, labels), axis=0)
    #     np.save("train_data",  self.data)
    #     np.save("train_label", self.labels)
    #     return

class TestDataset(Dataset):
    def __init__(self, data_path, batch_size):
        self.data_path = data_path
        self.batch = batch_size
        self.encoder = get_encoder_by_name("simple", 19)

        shape = self.encoder.shape()
        # self.data = np.zeros(np.insert(shape, 0, np.asarray([0])))
        # self.labels = np.zeros(0)
        self.data = []
        self.labels = []
        self.index = 0
        self.moves = 0
        # for idx, file_path in enumerate(data_path):
        #     if(idx % 1000 == 0):
        #         print("Data processing finished :", 100*idx/len(self.data_path), "% .")
        #     file = open(file_path).read()
        #     sgf = Sgf_game.from_string(file)
        #     self.moves +=num_total_moves(sgf)
    def __len__(self):
        # return self.moves
        return 500000

    def __getitem__(self, index):
        if len(self.data) < self.batch:
            self.index += 1
            temp_data, temp_label = get_data(self.data_path, self.index)
            self.data.extend(temp_data)
            self.labels.extend(temp_label)
        sample = self.data[0]
        label = self.labels[0]
        self.data.pop(0)
        self.labels.pop(0)
        return sample, label

def get_data(data_path, idx):
    encoder = get_encoder_by_name("simple", 19)

    shape = encoder.shape()
    data = []
    labels = []

    file_path = data_path[idx]
    file = open(file_path).read()
    sgf = Sgf_game.from_string(file)
    board, first_move_done = get_handicap(sgf)
    if board is None:
        board = Board(19, 19)

    cnt = 0
    for item in sgf.main_sequence_iter():
        color, move_tuple = item.get_move()
        point = None
        if color is not None:
            player = Color.BLACK if color == 'b' else Color.WHITE
            if move_tuple is not None:
                row, col = move_tuple
                point = Point(row+1, col+1)
            else:
                continue
            if first_move_done and point is not None:
                data.append(encoder.encode(board, color))
                # print(data)
                labels.append(encoder.encode_point(point))
                cnt += 1
            board.place_stone(player, point)
            first_move_done = True
    return data, labels


def get_handicap(sgf):
    go_board = Board(19, 19)
    first_move_done = False
    move = None
    # game_state = GameState.new_game(19)
    if sgf.get_handicap() is not None and sgf.get_handicap() != 0:
        for setup in sgf.get_root().get_setup_stones():
            for move in setup:
                row, col = move
                go_board.place_stone(Color.BLACK, Point(row + 1, col + 1))
        first_move_done = True
        # game_state = GameState(go_board, Color.WHITE, None, move)
    return go_board, first_move_done

def num_total_moves(sgf):
    total_moves = 0
    game_state, first_move_done = get_handicap(sgf)
    for item in sgf.main_sequence_iter():
        color, move = item.get_move()
        if color is not None:
            if first_move_done:
                total_moves += 1
            first_move_done = True
    return total_moves

if __name__ == '__main__':

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

    # value = (int)(len(train_data_path) / 8)
    # pool = []
    trainDataset = TrainDataSet(data_path=train_data_path, batch_size=10)
    # # trainDataset.get_data()
    dataloader = DataLoader(trainDataset, batch_size=4)
    cnt = 0
    for i, data in enumerate(dataloader):
        print(data)
        while(True):
            continue
    # for i in range (1000):
    #     get_data(train_data_path[0:1000], i)
    # for i in range(3):
    #     task = Process(target=get_data, args=(train_data_path[i*value : (i+1)*value], "train_data" + str(i)))
    #     pool.append(task)
    #     task.start()
    #
    # for task in pool:
    #     task.join()
    #
    # for i in range(4):
    #     task = Process(target=get_data, args=(train_data_path[(i+4)*value : (i+5)*value], "train_data" + str(i+4)))
    #     pool.append(task)
    #     task.start()
    #
    # for task in pool:
    #     task.join()