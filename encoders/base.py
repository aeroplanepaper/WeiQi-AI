import importlib

class Encoder:
    def name(self):
        raise NotImplementedError()

    """
    transform game state into a numpy array
    """
    def encode(self, board, player):
        raise NotImplementedError()

    """
    transform a point into index
    """
    def encode_point(self, point):
        raise NotImplementedError()

    """
    transform index into point
    """
    def decode_point_index(self, index):
        raise NotImplementedError()

    """
    the total number of points on the board
    """
    def num_points(self):
        raise NotImplementedError()

    """
        the shape of board after encoding
    """
    def shape(self):
        raise NotImplementedError()

def get_encoder_by_name(name, board_size):
    if isinstance(board_size, int):
            board_size = (board_size, board_size)
    module = importlib.import_module('encoders.' + name)
    constructor = getattr(module, 'create')
    return constructor(board_size)
