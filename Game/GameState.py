from Board import Board
from Move import Move
from Player import Color
import ZobristHash

import copy


class IllegalMoveError(Exception):
    pass


class GameState(object):

    def __init__(self, board: Board, next_player: Color, previous, move):
        """
        Initiation of a GameState at each round.
        :param board: Current board
        :param next_player: The next player who will take move.
        :param previous_state: The previous state of the game, as saving.
        :param move: The update move of this state.
        """
        self.board = board
        self.next_player = next_player
        self.previous_state = previous
        self.last_move = move
        if previous is not None:
            self.second_last_move = previous.last_move
        else:
            self.second_last_move = None
        if self.previous_state is None:
            self.previous_state = frozenset()
        else:
            self.previous_state = frozenset(
                previous.previous_state | {previous.next_player, previous.board.get_hash()})

    def apply_move(self, move: Move):
        if move.isPlay:
            try:
                self.does_move_self_captured(self.next_player, move)
            except IllegalMoveError:
                print("Self captured occurred! Please retry!")
                return None
            try:
                self.does_move_violate_ko(self.next_player, move)
            except IllegalMoveError:
                print("Ko fight occurred! Please retry!")
                return None
            next_board = copy.deepcopy(self.board)
            next_board.place_stone(self.next_player, move.point)
        else:
            next_board = self.board
        return GameState(next_board, self.next_player.opposite(), self, move)

    @classmethod
    def new_game(cls, board_size):
        assert isinstance(board_size, int), "Wrong type of board size"
        board = Board(board_size, board_size)
        return GameState(board, Color.BLACK, None, None)

    def isOver(self):
        if self.last_move is None:
            return False
        if self.last_move.isResign:
            return True
        if self.second_last_move is None:
            return False
        else:
            return self.last_move.isPass and self.second_last_move.isPass

    def does_move_self_captured(self, player, move):
        if not move.isPlay:
            return
        next_board = copy.deepcopy(self.board)
        next_board.place_stone(player, move.point)
        new_string = next_board.get_go_string(move.point)
        if new_string.num_liberties == 0:
            raise IllegalMoveError

    def does_move_violate_ko(self, player, move):
        if not move.isPlay:
            return
        next_board = copy.deepcopy(self.board)
        next_board.place_stone(player, move.point)
        next_situation = (player.opposite, next_board.get_hash())
        if next_situation in self.previous_state:
            raise IllegalMoveError
