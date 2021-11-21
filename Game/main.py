from GUI import GUI
from Board import  Board
from agent.Agent import AI,Human
import Player
from GameState import GameState
import pygame

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    game = GameState.new_game(19)

    gui = GUI(game.board)
    player1 = AI()
    player2 = Human()
    running = True
    while running:
        # if game.next_player == Player.Color.BLACK:



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
