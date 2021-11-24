import pygame
from Board import Point

def place_stone_control():
    print(pygame.event.get())
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                x, y = event.pos
                row = round((x - 40) / 40 + 1)
                col = round((y - 40) / 40 + 1)
                return Point(row, col)
            else:
                pass
