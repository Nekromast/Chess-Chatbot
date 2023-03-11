import torch


class ChessAI:
    def __init__(self, model_path):
        self.model = torch.load(model_path)

    def make_move(self):
        #best_move = self.model.predict(board_state)
        best_move = 0
        # Hier Reinforcement Learning
        
        return best_move



