import enum

class Player(enum.Enum):
    AI = 1
    HUMAN = 2

    def next(self):
        return Player.AI if self == Player.HUMAN else Player.AI

class Color(enum.Enum):
    BLACK = 1
    WHITE = 2

    def opposite(self):
        return Color.BLACK if self == Color.WHITE else Color.WHITE
