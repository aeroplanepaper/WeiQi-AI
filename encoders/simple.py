import numpy as np

from encoders.base import Encoder
from Game.Move import Move
from Game.Board import Point
from Game.Player import Color


class SimpleEncoder(Encoder):
    def __init__(self, board_size):
        """
        Args:
            board_size: tuple of (width, height)
        """
        self.board_width, self.board_height = board_size
        # 0 - 3. black stones with 1, 2, 3, 4+ liberties
        # 4 - 7. white stones with 1, 2, 3, 4+ liberties
        # 8. black plays next
        # 9. white plays next
        # 10. move would be illegal due to ko
        self.num_planes = 10

    def name(self):
        return 'simple'

    def encode(self, board, color):
        board_tensor = np.zeros(self.shape())
        if color == Color.BLACK:
            board_tensor[8] = 1
        else:
            board_tensor[9] = 1
        for r in range(self.board_height):
            for c in range(self.board_width):
                p = Point(row=r + 1, col=c + 1)
                go_string = board.get_go_string(p)

                if go_string is None:
                    continue
                    # if game_state.does_move_violate_ko(game_state.next_player,
                    #                                    Move.play(p)):
                    #     board_tensor[10][r][c] = 1
                else:
                    liberty_plane = min(4, go_string.num_liberties) - 1
                    if go_string.color == Color.WHITE:
                        liberty_plane += 4
                    board_tensor[liberty_plane][r][c] = 1

        return board_tensor

    def encode_point(self, point):
        """Turn a board point into an integer index."""
        # Points are 1-indexed
        return self.board_width * (point.row - 1) + (point.col - 1)

    def decode_point_index(self, index):
        """Turn an integer index into a board point."""
        row = index // self.board_width
        col = index % self.board_width
        return Point(row=row + 1, col=col + 1)

    def num_points(self):
        return self.board_width * self.board_height

    def shape(self):
        return self.num_planes, self.board_height, self.board_width


def create(board_size):
    return SimpleEncoder(board_size)
