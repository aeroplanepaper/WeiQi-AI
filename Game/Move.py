
import enum
import copy


class Move(object):
    def __init__(self, point=None, isPass=False, isResign=False):
        """
        Three types of moves at each round.
        :param point: The point to place a stone, is not None only when the Move is play.
        :param isPass: Is this move a pass.
        :param isResign: Is this move a resign.
        """
        assert (point is not None) ^ isPass ^ isResign, "Wrong Move Type"
        self.point = point
        self.isPass = isPass
        self.isResign = isResign
        self.isPlay = (point is not None)

    @classmethod
    def play(cls, point):
        return Move(point=point)

    @classmethod
    def passTurn(cls):
        return Move(isPass=True)

    @classmethod
    def resign(cls):
        return Move(isResign=True)
