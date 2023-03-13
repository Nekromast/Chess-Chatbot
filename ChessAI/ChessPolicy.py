import gym
from gym_chess import ChessEnvV2
import torch
import random
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import namedtuple

import matplotlib.pyplot as plt
from IPython import display
from tqdm.notebook import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ChessPolicyNet(nn.Module):
    def __init__(self):
        super(ChessPolicyNet, self).__init__()
        self.conv1 = nn.Conv2d(12, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 64)
        self.fc4_start = nn.Linear(64, 64)
        self.fc4_end = nn.Linear(64, 64 * 64)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x_start = self.fc4_start(x)
        x_end = self.fc4_end(x).view(1, 64, 64)

        return x_start, x_end


class ChessPolicy:
    def __init__(self):
        self.net = ChessPolicyNet().to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.03)
        self.mean_reward = None
        self.games = 0
        self.gamma = 0.99
        self.eps = np.finfo(np.float32).eps.item()
        self.SavedAction = namedtuple('SavedAction', ['start_prob', 'end_prob'])

    def __call__(self, observation, possible_moves):
        board = np.array(observation['board'])
        one_hot_board = np.zeros((8, 8, 12))
        for i in range(8):
            for j in range(8):
                piece = board[i][j]
                if piece != 0:
                    if piece > 0:
                        one_hot_board[i][j][abs(piece) - 1] = 1
                    else:
                        one_hot_board[i][j][abs(piece) + 5] = 1
        one_hot_board = np.transpose(one_hot_board, (2, 0, 1))
        x = torch.from_numpy(one_hot_board).float().unsqueeze(0)
        x = x.to(device)
        x_start, x_end = self.net(x)

        # Get starting and end positions that are legal
        legal_starting_moves = torch.zeros((8, 8)).cuda()
        legal_moves = torch.zeros((8, 8, 8, 8)).cuda()
        for move in possible_moves:
            if len(move) == 2:
                x, y = move[0]
                x2, y2 = move[1]
                legal_starting_moves[x][y] = 1
                legal_moves[x][y][x2][y2] = 1

        # Filter out illegal moves
        x_start_2d = x_start[0].view(8, 8)
        x_start_2d = legal_starting_moves * x_start_2d
        x_end_4d = x_end[0].view(8, 8, 8, 8)
        x_end_4d = legal_moves * x_end_4d

        # Reshape back into 1D Tensors
        x_start = x_start_2d.view(1, 64)
        x_end = x_end_4d.view(1, 64, 64)

        # Softmax starting positions
        non_zero_mask = (x_start[0] != 0)  # boolean mask of non-zero elements
        non_zero_values = x_start[0][non_zero_mask]  # extract non-zero values
        softmaxed_values = torch.softmax(non_zero_values, dim=0)  # apply softmax to non-zero values
        start_result = torch.zeros_like(x_start)  # create a tensor with the same shape as probs, filled with zeros
        start_result[0][non_zero_mask] = softmaxed_values  # put softmaxed non-zero values back into result tensor

        # Sample starting positions
        m_start = Categorical(start_result)
        start_tensor = m_start.sample(sample_shape=torch.Size([]))
        start_item = start_tensor.item()
        start_pos = (start_item // 8, start_item % 8)

        # Softmax end positions
        non_zero_mask = (x_end[0][start_item] != 0)  # boolean mask of non-zero elements
        non_zero_values = x_end[0][start_item][non_zero_mask]  # extract non-zero values
        softmaxed_values = torch.softmax(non_zero_values, dim=0)  # apply softmax to non-zero values
        end_result = torch.zeros_like(x_end)  # create a tensor with the same shape as probs, filled with zeros
        end_result[0][start_item][
            non_zero_mask] = softmaxed_values  # put softmaxed non-zero values back into result tensor

        # Sample end positions
        m_end = Categorical(end_result[0][start_item])
        end_tensor = m_end.sample(sample_shape=torch.Size([]))
        end_item = end_tensor.item()
        end_pos = (end_item // 8, end_item % 8)

        self.memory.append(self.SavedAction(m_start.log_prob(start_tensor), m_end.log_prob(end_tensor)))
        return start_pos, end_pos

    def init_game(self, observation, possible_moves):
        self.memory = []
        self.rewards = []
        self.total_reward = 0

    def update(self, observation, reward, terminated, truncated, info, status):
        self.total_reward += reward
        self.rewards.append(reward)
        if terminated:
            self.games += 1
            if self.mean_reward is None:
                self.mean_reward = self.total_reward
            else:
                self.mean_reward = self.mean_reward * 0.95 + self.total_reward * (1.0 - 0.95)

            # calculate discounted reward and make it normal distributed
            discounted = []
            R = 0
            for r in self.rewards[::-1]:
                R = r + self.gamma * R
                discounted.insert(0, R)
            discounted = torch.Tensor(discounted)
            discounted = (discounted - discounted.mean()) / (discounted.std() + self.eps)

            policy_losses = []
            for mem, discounted_reward in zip(self.memory, discounted):
                start_prob, end_prob = mem
                policy_losses.append(-((start_prob + end_prob) * discounted_reward))
            # TODO if zero grads makes probs tensors NaN later on
            # self.optimizer.zero_grad()
            loss = torch.stack(policy_losses).sum()
            loss.backward()
            self.optimizer.step()

            if self.games % 50 == 0:
                self.save(f"models/model_games_{self.games}.pt")

    def load(self, PATH):
        checkpoint = torch.load(PATH)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.games = checkpoint['games']
        self.mean_reward = checkpoint['mean_reward']

    def save(self, PATH):
        torch.save({
            'games': self.games,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'mean_reward': self.mean_reward}, PATH)
