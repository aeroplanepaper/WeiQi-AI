from GUI import GUI
from Board import Board
from Board import Point
from Player import Player
from GameState import GameState
from Move import Move
import Control
import pygame

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    game = GameState.new_game(19)

    gui = GUI()
    player1 = Player.HUMAN
    player2 = Player.AI
    running = True
    current_player = player1
    while running:
        if current_player == Player.AI:
            pass
        else:
            point = Control.place_stone_control()
            if point is None:
                pygame.quit()
            move = Move(point)
            print(point)
            game = game.apply_move(move)
            gui.update(game.board)




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
