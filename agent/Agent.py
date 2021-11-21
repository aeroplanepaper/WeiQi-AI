class Agent:
    def __init__(self):
        pass

    def select_move(self, game_state):
        raise NotImplementedError()

class AI(Agent):
    def select_move(self, game_state):
        pass

class Human(Agent):
    def select_move(self, game_state):
        pass
    def