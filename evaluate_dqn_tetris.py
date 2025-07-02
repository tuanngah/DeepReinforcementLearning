import torch
import numpy as np
import matplotlib.pyplot as plt
from tetris_env_2 import TetrisEnv

from train_dqn_tetris import QNetwork  
from train_double_dqn_tetris import QNetwork
from train_dueling_dqn_tetris import DuelingQNetwork
from train_a2c_tetris import ActorCritic

import time

MODEL_PATH = 'dqn_double_logged_5_new.pth'  
NUM_EPISODES = 10
RENDER_DELAY = 0.05  # seconds between frames

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Loading model on device: {device}')

# Initialize environment and network
env = TetrisEnv()
in_shape = env.observation_space.shape
n_actions = env.action_space.n
net = QNetwork(in_shape, n_actions).to(device)
net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
net.eval()

# Evaluate
episode_rewards = []

for ep in range(NUM_EPISODES):
    state = env.reset()
    state = np.array(state, dtype=np.float32) / 255.0
    total = 0.0
    done = False
    print(f"=== Episode {ep+1} ===")
    while not done:
        # Render environment to screen
        env.render(mode='human')
        time.sleep(RENDER_DELAY)

        # Greedy action
        with torch.no_grad():
            s_v = torch.tensor([state], device=device)
            q_vals = net(s_v)
            action = int(q_vals.argmax(dim=1).item())
        next_state, reward, done, _ = env.step(action)
        state = np.array(next_state, dtype=np.float32) / 255.0
        total += reward
    episode_rewards.append(total)
    print(f'Episode {ep+1}: Reward = {total:.2f}')

env.close()

# Print summary
print(f'Average Reward over {NUM_EPISODES} episodes: {np.mean(episode_rewards):.2f}')
print(f'Max Reward: {np.max(episode_rewards):.2f}')
print(f'Min Reward: {np.min(episode_rewards):.2f}')