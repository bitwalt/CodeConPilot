"""A General Artificial Intelligence made in Pytorch"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def initialize_weights(layer):
    """Initializes the weights of the layer by sampling from a normal distribution
    with mean 0 and standard deviation 1/sqrt(n) where n is the number of 
    weights in the layer. Used to initialize the weights of the network"""
    layer.weight.data.normal_(mean=0, std=1/np.sqrt(layer.weight.data.size()[0]))
    layer.bias.data.fill_(0)
      

class Trainer(object):

    def __init__(self, state_size, action_size, seed):
        """Initialize the trainer class.
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

class GeneralizedQNetwork(QNetwork):

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64): 
        """Initialize the agent and the networks"""
        super(GeneralizedQNetwork, self).__init__(state_size, action_size, seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.optimizer = optim.Adam(self.parameters(), lr=LR)
        self.loss = nn.MSELoss()
        self.fc1.weight.data.uniform_(*initialize_weights(self.fc1))
        self.fc2.weight.data.uniform_(*initialize_weights(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.to(device)
        
                 
    def update(self, input_data):
        """Updates the network"""
        self.optimizer.zero_grad()
        state, action, reward, next_state, done = input_data
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(device)
        reward = torch.from_numpy(np.array([reward])).float().unsqueeze(0).to(device)
        action = torch.from_numpy(np.array([action])).long().unsqueeze(0).to(device)
        done = torch.from_numpy(np.array([done])).float().unsqueeze(0).to(device)
        Q_values = QNetwork(state)
        Q_values_next = QNetwork(next_state)
        Q_value = Q_values.gather(1, action)
        Q_value_next = torch.max(Q_values_next, dim=1)[0]
        target = reward + GAMMA*Q_value_next*(1-done)
        loss = F.mse_loss(Q_value, target)
        loss.backward()
        QNetwork.optimizer.step()
        return loss.item()
    
    def train(self, input_data):
        """Trains the network"""
        
        dataset = []
        for state, action, reward, next_state, done in input_data:
            dataset.append([state, action, reward, next_state, done])
        dataset = np.array(dataset)
        np.random.shuffle(dataset)
        batch_size = int(BATCH_SIZE/len(input_data))
        batches = [dataset[x:x+batch_size] for x in range(0, len(dataset), batch_size)]
        loss = 0
        for batch in batches:
            loss += self.update(batch)
        return loss 
    
        