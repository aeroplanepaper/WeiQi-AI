from GUI import GUI
from Board import Board
from Board import Point
from Player import Player
import copy
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
            game_backUp = copy.deepcopy(game)
            game_backUp = game_backUp.apply_move(move)
            if game_backUp is None:
                continue
            game = game_backUp
            gui.update(game.board)
            # current_player = current_player.next()




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
