# ============================
# Imports
# ============================
import copy
import imp 

import torch
from torch import nn
from torch.functional import F
import numpy as np
import warnings
np.warnings = warnings  # quiet NumPy warnings
import gymnasium as gym
import os

import experiment_settings
from tqdm import tqdm
import random
import register_envs

# ============================
# Device Configuration
# ============================
# # if GPU is to be used
# device = torch.device(
#     "cuda" if torch.cuda.is_available() else
#     "mps" if torch.backends.mps.is_available() else
#     "cpu"
# )

# Force GPU use
device = torch.device("cuda")

print(torch.cuda.is_available())
# quit()

# ============================
# Load experiment settings
# ============================
EXP_NAME = "falco_test"

imp.reload(experiment_settings)
settings = experiment_settings.get_settings(EXP_NAME)

DATA_FP = settings['file_path']
ACE_FP  = settings['ace_path']

# ============================
# Custom environment wrapper for preprocessing
# ============================
class PreProcessEnv(gym.Wrapper):

    def __init__(self, env, fp=None, ace_fp=None):
        gym.Wrapper.__init__(self, env)

    def step(self, action, fp, episode):
        self.update_episode(episode)  # dynamic config for this episode
        action = action.item()
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Convert to torch tensors
        obs = torch.from_numpy(obs).unsqueeze(0).float()
        reward = torch.tensor(reward).view(-1, 1)
        terminated = torch.tensor(terminated).view(-1, 1)
        truncated = torch.tensor(truncated).view(-1, 1)

        return obs, reward, terminated, truncated, info

    def reset(self):
        obs = self.env.reset()
        obs = torch.from_numpy(obs).unsqueeze(0).float()
        return obs

# ============================
# Instantiate and wrap environment
# ============================
env = gym.make("ClimateControl-v0", fp=DATA_FP, ace_fp=ACE_FP)
env = PreProcessEnv(env, fp=DATA_FP)

# ============================
# DQN Model Definition
# ============================
class DQNetworkModel(nn.Module):
    def __init__(self, in_channels, out_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out_classes),
        )

    def forward(self, x):
        return self.layers(x)

# ============================
# Instantiate Q-network and target Q-network
# ============================

q_network = DQNetworkModel(env.observation_space.shape[0], env.action_space.n).to(device)

print(env.action_space.n)
target_q_network = copy.deepcopy(q_network).to(device).eval()

# ============================
# Replay Memory Class
# ============================
class ReplayMemory:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def insert(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def can_sample(self, batch_size):
        return len(self.memory) >= batch_size * 10

    def sample(self, batch_size):
        assert self.can_sample(batch_size)
        transitions = random.sample(self.memory, batch_size)
        batch = zip(*transitions)
        return [torch.cat([item for item in items]) for items in batch]

# ============================
# Epsilon-greedy policy
# ============================
def policy(state, epsilon):
    if torch.rand(1) < epsilon:
        return torch.randint(env.action_space.n, (1, 1))
    else:
        av = q_network(state).detach()
        return torch.argmax(av, dim=-1, keepdim=True)

# ============================
# DQN Training Loop
# ============================
def dqn_training(
    q_network: DQNetworkModel,
    policy,
    episodes,
    alpha=0.0001,
    batch_size=128,
    gamma=0.99,
    epsilon=1,
):
    optim = torch.optim.AdamW(q_network.parameters(), lr=alpha)
    memory = ReplayMemory()
    stats = {'MSE Loss': [], 'Returns': [], 'Reward': [], 'Action': []}
    
    for episode in tqdm(range(1, episodes + 1)):
        # Ensure directory exists for this episode
        config_dir = '/home/nicojg/ace/configs/' + EXP_NAME + '/episode' + str(episode)
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)

        state = env.reset()
        truncated, terminated = False, False
        ep_return = 0

        while not truncated and not terminated:
            action = policy(state, epsilon)
            stats['Action'].append(action.item())

            next_state, reward, truncated, terminated, _ = env.step(action, DATA_FP, episode)
            stats['Reward'].append(reward.item())

            memory.insert([state, action, reward, truncated, terminated, next_state])

            if memory.can_sample(batch_size):
                # Sample from replay memory
                state_b, action_b, reward_b, truncated_b, terminated_b, next_state_b = memory.sample(batch_size)

                qsa_b = q_network(state_b).gather(1, action_b)
                next_qsa_b = target_q_network(next_state_b)
                next_qsa_b = torch.max(next_qsa_b, dim=-1, keepdim=True)[0]

                # Q-learning update
                target_b = reward_b + ~(truncated_b + terminated_b) * gamma * next_qsa_b
                loss = F.mse_loss(qsa_b, target_b)

                q_network.zero_grad()
                loss.backward()
                optim.step()

                stats['MSE Loss'].append(loss)

            state = next_state
            ep_return += reward.item()

        stats['Returns'].append(ep_return)
        epsilon = max(0, epsilon - 1 / 10000)

        # Periodically update the target network
        if episode % 10 == 0:
            target_q_network.load_state_dict(q_network.state_dict())

    return stats

# ============================
# Run Training
# ============================
stats = dqn_training(q_network, policy, 10)

# ============================
# Output Final Stats
# ============================
print(stats['Action'])
print(stats['Reward'])

# ============================
# Optional: run trained policy in human-rendered env
# ============================
# env = gym.make("CartPole-v1", render_mode="human")
# env = PreProcessEnv(env)
# q_network.eval()
# for i in range(100):
#     state = env.reset()
#     terminated, truncated = False, False
#     while not terminated and not truncated:
#         with torch.inference_mode():
#             action = torch.argmax(q_network(state.to(device)))
#             state, reward, terminated, truncated, info = env.step(action)
