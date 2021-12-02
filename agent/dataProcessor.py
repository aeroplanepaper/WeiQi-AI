import os
import numpy as np
from torch.utils.data import DataLoader, Dataset

from gosgf.sgf import Sgf_game
from Game.GameState import GameState
from Game.Board import Board,Point
from Game.Move import Move
from Game.Player import Color
from encoders.base import get_encoder_by_name


class TrainDataSet(Dataset):
    def __init__(self, data_path, batch_size):
        self.data_path = data_path
        self.batch = batch_size
        self.encoder = get_encoder_by_name("oneplane", 19)

        shape = self.encoder.shape()
        self.data = np.zeros(np.insert(shape, 0, np.asarray([0])))
        self.labels = np.zeros(0)
        self.index = 0
        self.percent = 0


    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, index):
        if self.data.shape[0] < self.batch:
            self.get_data()

        sample = self.data[0]
        label = self.labels[0]
        self.data = np.delete(self.data, 0, 0)
        self.labels = np.delete(self.labels, 0, 0)
        return sample, label

    def get_data(self):
        file_path = self.data_path[self.index]
        temp = int(100*(self.index + 1)/self.__len__())
        if self.percent - temp != 0:
            self.percent = temp
            print(self.percent)
        self.index += 1
        if not file_path.endswith(".sgf"):
            raise ValueError(file_path + "is not sgf file!")
        file = open(file_path).read()
        sgf = Sgf_game.from_string(file)
        game_state, first_move_done = get_handicap(sgf)
        shape = self.encoder.shape()
        total_move = num_total_moves(sgf)
        # print(total_move)
        feature_shape = np.insert(shape, 0, np.asarray([total_move]))
        # print(feature_shape)
        features = np.zeros(feature_shape)
        labels = np.zeros(total_move)
        cnt = 0
        for item in sgf.main_sequence_iter():
            color, move_tuple = item.get_move()
            point  = None
            if color is not None:
                if move_tuple is not None:
                    row, col = move_tuple
                    point = Point(row+1, col+1)
                    move = Move(point)
                else:
                    move = Move.passTurn()
                if first_move_done and point is not None:
                    features[cnt] = self.encoder.encode(game_state)
                    labels[cnt] = self.encoder.encode_point(point)
                    cnt += 1
                game_state = game_state.apply_move(move)
                first_move_done = True
        self.data = np.concatenate((self.data, features), axis=0)
        self.labels = np.concatenate((self.labels, labels), axis=0)
        return

class TestDataset(Dataset):
    def __init__(self, data_path, batch_size):
        self.data_path = data_path
        self.batch = batch_size
        self.encoder = get_encoder_by_name("oneplane", 19)

        shape = self.encoder.shape()
        self.data = np.zeros(np.insert(shape, 0, np.asarray([0])))
        self.labels = np.zeros(0)
        self.index = 0


    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, index):
        if self.data.shape[0] < self.batch:
            self.get_data()

        sample = self.data[0]
        label = self.labels[0]
        self.data = np.delete(self.data, 0, 0)
        self.labels = np.delete(self.labels, 0, 0)
        return sample, label

    def get_data(self):
        file_path = self.data_path[self.index]
        self.index += 1
        if not file_path.endswith(".sgf"):
            raise ValueError(file_path + "is not sgf file!")
        file = open(file_path).read()
        sgf = Sgf_game.from_string(file)
        game_state, first_move_done = get_handicap(sgf)
        shape = self.encoder.shape()
        total_move = num_total_moves(sgf)
        # print(total_move)
        feature_shape = np.insert(shape, 0, np.asarray([total_move]))
        # print(feature_shape)
        features = np.zeros(feature_shape)
        labels = np.zeros(total_move)
        cnt = 0
        for item in sgf.main_sequence_iter():
            color, move_tuple = item.get_move()
            point  = None
            if color is not None:
                if move_tuple is not None:
                    row, col = move_tuple
                    point = Point(row+1, col+1)
                    move = Move(point)
                else:
                    move = Move.passTurn()
                if first_move_done and point is not None:
                    features[cnt] = self.encoder.encode(game_state)
                    labels[cnt] = self.encoder.encode_point(point)
                    cnt += 1
                game_state = game_state.apply_move(move)
                first_move_done = True
        self.data = np.concatenate((self.data, features), axis=0)
        self.labels = np.concatenate((self.labels, labels), axis=0)
        return


def get_handicap(sgf):
    go_board = Board(19, 19)
    first_move_done = False
    move = None
    game_state = GameState.new_game(19)
    if sgf.get_handicap() is not None and sgf.get_handicap() != 0:
        for setup in sgf.get_root().get_setup_stones():
            for move in setup:
                row, col = move
                go_board.place_stone(Color.BLACK, Point(row + 1, col + 1))
        first_move_done = True
        game_state = GameState(go_board, Color.WHITE, None, move)
    return game_state, first_move_done

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

