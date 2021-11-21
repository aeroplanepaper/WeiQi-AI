import pygame
import time
from Board import Board
from Board import Point
from Player import Color

black_color = [0, 0, 0]
white_color = [255, 255, 255]

class GUI(object):
    def __init__(self, board: Board):
        self.board = board
        pygame.init()
        pygame.display.set_caption("围棋")
        self.screen = pygame.display.set_mode((800, 820))
        self.screen.fill([125, 95, 24])
        for h in range(1, 20):
            pygame.draw.line(self.screen, black_color, [40, h*40], [760, h*40], 1)
            pygame.draw.line(self.screen, black_color, [h*40, 40], [h*40, 760], 1)
        pygame.draw.rect(self.screen, black_color, [36, 36, 728, 728], 3)

        pygame.draw.circle(self.screen, black_color, [400, 400], 4, 0)
        pygame.draw.circle(self.screen, black_color, [400, 160], 4, 0)
        pygame.draw.circle(self.screen, black_color, [160, 400], 4, 0)
        pygame.draw.circle(self.screen, black_color, [640, 400], 4, 0)
        pygame.draw.circle(self.screen, black_color, [400, 640], 4, 0)
        pygame.draw.circle(self.screen, black_color, [160, 160], 4, 0)
        pygame.draw.circle(self.screen, black_color, [640, 640], 4, 0)
        pygame.draw.circle(self.screen, black_color, [160, 640], 4, 0)
        pygame.draw.circle(self.screen, black_color, [640, 160], 4, 0)
        pygame.display.flip()

    def update(self):
        for row in range(self.board.num_rows):
            for col in range(self.board.num_cols):
                point = Point(row, col)
                stone = self.board.get_stone(point)
                if stone is not None:
                    pos = [40*row, 40*col]
                    pygame.draw.circle(self.screen, white_color if stone == Color.WHITE else black_color, pos, 18, 0)
        pygame.display.flip()


if __name__ == '__main__':
    board = Board(19, 19)
    gui = GUI(board)
    gui.board.place_stone(Color.BLACK, Point(4,4))
    time.sleep(1)
    gui.update()
    running = True
    while running:
        time.sleep(20)
        running = False
