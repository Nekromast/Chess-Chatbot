import torch
import gym
from gym_chess import ChessEnvV2
from ChessPolicy import ChessPolicy


class ChessAI:
    def __init__(self):
        self.policy = None
        self.env = None
        self.vs_ai = True
        self.rewards = {}
        self.observation = None
        self.color = None

    def start_game(self, color, vs_ai=True):
        self.env = ChessEnvV2(color, opponent="none", log=False)
        self.observation = self.env.reset()
        self.vs_ai = vs_ai
        self.rewards = {"white": 0, "black": 0}
        self.color = color
        if vs_ai:
            self.policy = ChessPolicy()
            self.policy.init_game(self.observation, self.env.possible_moves)

    def make_move(self, move):
        action = self.env.move_to_action(move)
        self.make_action(action)

        # AI moves
        if self.vs_ai:
            move = self.policy(self.observation, self.env.possible_moves)
            action = self.env.move_to_action(move)
            self.make_action(action)

    def make_action(self, action):
        observation, step_reward, done, info = self.env.step(action)
        self.rewards[f"{self.env.current_player}"] += step_reward
        if done:
            self.env.reset()
            self.rewards = {"white": 0, "black": 0}
            return done

    def get_board(self):
        return self.env.render

    def get_possible_moves(self):
        return self.env.possible_moves

    def get_reward(self):
        if self.vs_ai:
            return self.rewards[f"{self.color}"]
        else:
            return self.rewards[f"{self.env.current_player}"]

    def quit_game(self):
        action = self.env.move_to_action("RESIGN")
        observation, step_reward, done, info = self.env.step(action)
        self.env.close()
