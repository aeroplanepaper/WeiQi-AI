import enum


class Color(enum.Enum):
    BLACK = 1
    WHITE = 2

    def opposite(self):
        return Color.BLACK if self == Color.WHITE else Color.WHITE
