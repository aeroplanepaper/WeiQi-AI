import numpy as np
from encoders.base import Encoder
from Game.Board import Point
from numba import jit

class OnePlaneEncoder(Encoder):
    def __init__(self, board_size):
        self.board_width, self.board_height = board_size
        self.num_planes = 1

    def name(self):
        return 'oneplane'

    """
    if color of point = next player, matrix_index = -1
    if color of point != next player, matrix_index = -1
    else matrix index = 0
    """
    def encode(self, board, player):
        board_matrix = np.zeros(self.shape())
        # next_player = game_state.next_player
        for r in range(self.board_height):
            for c in range(self.board_width):
                p = Point(row=r + 1, col=c + 1)
                color = board.get_stone(p)
                if color is None:
                    continue
                if color == player:
                    board_matrix[0, r, c] = 1
                else:
                    board_matrix[0, r, c] = -1
        return board_matrix

    def encode_point(self, point):
        return self.board_width * (point.row - 1) + point.col - 1

    def decode_point_index(self, index):
        row = index // self.board_width
        col = index % self.board_width
        return Point(row=row + 1, col=col + 1)

    def num_points(self):
        return self.board_width * self.board_height

    def shape(self):
        return self.num_planes, self.board_height, self.board_width

def create(board_size):
    return OnePlaneEncoder(board_size)
