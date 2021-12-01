
from collections import namedtuple
from Game.Player import Color



class Point(namedtuple('Point', 'row, col')):
    """
    The point of the board.
    """

    def neighbors(self):
        """

        :return:The list of the neighbored points.
        """
        return [
            Point(self.row - 1, self.col),
            Point(self.row + 1, self.col),
            Point(self.row, self.col + 1),
            Point(self.row, self.col - 1),
        ]


class GoString(object):
    def __init__(self, color: Color, stones, liberties):
        """
        A combination of chained stones.
        :param color:  The color of the String.
        :param stones: The set of the stones.
        :param liberties: The set of the points that are liberties.
        """
        self.color = color
        self.stones = frozenset(stones)
        self.liberties = frozenset(liberties)

    def remove_liberty(self, point: Point):
        """
        This method is used when a new stone is placed.
        :param point: The target point to remove its liberty.
        """
        new_liberties = self.liberties - {point}
        return GoString(self.color, self.stones, new_liberties)

    def add_liberty(self, point: Point):
        """
        This method is used when a new stone is removed.
        :param point: The target point to add its liberty.
        """
        new_liberties = self.liberties | {point}
        return GoString(self.color, self.stones, new_liberties)

    def merge(self, goString):
        """
        This method is used when two strings are neighbored.
        :param goString: the target String going to merge with.
        :return: The merged new String.
        """
        assert goString.color == self.color, "Wrong Merge, color not match"
        combined_stones = self.stones | goString.stones
        # print(combined_stones)
        return GoString(
            self.color,
            combined_stones,
            (self.liberties | goString.liberties) - combined_stones
        )

    @property
    def num_liberties(self):
        """
        This method used when checking a GoString's liberties.
        :return: The liberties of a String.
        """
        return len(self.liberties)

    def __eq__(self, other):
        """
        Overwrite of the __eq__ method.
        :param other: The GoString going to compare.
        :return: The equivalence of two GoString.
        """
        return isinstance(other, GoString) and \
               self.color == other.color and \
               self.stones == other.stones and \
               self.liberties == other.liberties


class Board(object):

    def __init__(self, num_rows: int, num_cols: int):
        """
        The initiation of the board.
        :param num_rows: Number of rows of the board.
        :param num_cols: Number of columns of the board.
        """
        import Game.ZobristHash
        self.num_rows = num_rows
        self.num_cols = num_cols
        "_grid: The status of current board, every stone functioned to its corresponding GoString"
        self._grid = {}
        self._hash = Game.ZobristHash.EMPTY_BOARD

    def is_on_grid(self, point: Point) -> bool:
        """
        This method check if a point is legal, NOT A STONE.
        :param point: A point to check.
        :return: Whether the point is legal on this board or not.
        """
        return 1 <= point.row <= self.num_rows and 1 <= point.col <= self.num_cols

    def get_stone(self, point: Point):
        """
        This method get the the stone on the target point.
        :param point: A point to get the color of the stone on it.
        :return: The color of the stone on this point. RETURN NONE IF NO STONE ON IT!
        """
        string = self._grid.get(point)
        if string is None:
            return None
        return string.color

    def get_go_string(self, point: Point):
        """
        This method get the GoString of the stone on the target point.
        :param point: A point to get the corresponding GoString of the stone on it.
        :return: The GoString
        """
        string = self._grid.get(point)
        return string

    def _replace_string(self, new_string):
        for point in new_string.stones:
            self._grid[point] = new_string

    def _remove_string(self, string: GoString):
        """
        This method remove the target GoString, and update the liberties of adjacent GoString.
        :param string: The String trying to remove
        """
        import Game.ZobristHash
        # print(string.stones)
        for removed_point in string.stones:
            for neighbor in removed_point.neighbors():
                if self.is_on_grid(neighbor):
                    adjacent_string = self._grid.get(neighbor)
                    if adjacent_string is None:
                        continue
                    if adjacent_string is not string:
                        self._replace_string(adjacent_string.add_liberty(removed_point))
            self._grid[removed_point] = None
            self._hash ^= Game.ZobristHash.HASH_CODE[removed_point, string.color]

    def place_stone(self, playerColor: Color, point: Point):
        """
            This method place a stone on the target point. It will simultaneously update the the state of the
        board, including updating the liberties of the adjacent stones, and removing the dead GoString.
        :param playerColor:  The color of the current player.
        :param point: The target point to place the stone.
        """
        assert self.is_on_grid(point), "Point is not on the board!"
        assert self._grid.get(point) is None, "Stone already exist, illegal move!"
        import Game.ZobristHash
        adjacent_same_color = []
        adjacent_opposite_color = []
        liberties = []

        for neighbor in point.neighbors():
            if not self.is_on_grid(neighbor):
                continue
            neighbor_string = self._grid.get(neighbor)
            if neighbor_string is None:
                liberties.append(neighbor)
            elif neighbor_string.color == playerColor:
                if neighbor_string not in adjacent_same_color:
                    adjacent_same_color.append(neighbor_string)
            else:
                if neighbor_string not in adjacent_opposite_color:
                    adjacent_opposite_color.append(neighbor_string)

        new_string = GoString(playerColor, [point], liberties)
        for same_color_string in adjacent_same_color:
            new_string = new_string.merge(same_color_string)
            # print("merge")
        for new_string_point in new_string.stones:
            self._grid[new_string_point] = new_string
        self._hash ^= Game.ZobristHash.HASH_CODE[point, playerColor]

        for opposite_color_string in adjacent_opposite_color:
            replacement_string = opposite_color_string.remove_liberty(point)
            if replacement_string.num_liberties == 0:
                self._remove_string(opposite_color_string)
                # print('remove')
            else:
                self._replace_string(replacement_string)

    def get_hash(self):
        return self._hash
