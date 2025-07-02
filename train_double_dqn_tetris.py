import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tetris_env_2 import TetrisEnv
import matplotlib.pyplot as plt

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training Double DQN on device: {device}")

def conv2d_size(size, kernel, stride):
    return (size - (kernel - 1) - 1) // stride + 1

# Q-network (CNN)
class QNetwork(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(QNetwork, self).__init__()
        c, h, w = input_shape
        self.net = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self._feature_size(h, w), 512), nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _feature_size(self, h, w):
        h1 = conv2d_size(h, 8, 4)
        w1 = conv2d_size(w, 8, 4)
        h2 = conv2d_size(h1, 4, 2)
        w2 = conv2d_size(w1, 4, 2)
        h3 = conv2d_size(h2, 3, 1)
        w3 = conv2d_size(w2, 3, 1)
        return h3 * w3 * 64

    def forward(self, x):
        return self.net(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=bool)
        )
    def __len__(self):
        return len(self.buffer)

# Training loop
def train():
    # Hyperparameters
    capacity = 100_000
    batch_size = 128
    gamma = 0.98
    lr = 5e-4
    sync_target_steps = 5_000
    max_frames = 500_000
    eps_start, eps_final, eps_decay = 1.0, 0.1, 100_000

    env = TetrisEnv()
    buffer = ReplayBuffer(capacity)

    input_shape = env.observation_space.shape
    n_actions = env.action_space.n

    online_net = QNetwork(input_shape, n_actions).to(device)
    target_net = QNetwork(input_shape, n_actions).to(device)
    target_net.load_state_dict(online_net.state_dict())

    optimizer = optim.Adam(online_net.parameters(), lr=lr)

    state = env.reset()
    state = np.array(state, dtype=np.float32) / 255.0

    frame_idx = 0
    episode = 0
    episode_reward = 0
    episode_rewards = []

    while frame_idx < max_frames:
        eps = eps_final + (eps_start - eps_final) * max(0, (eps_decay - frame_idx) / eps_decay)
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_v = torch.tensor([state], device=device)
                action = int(online_net(state_v).argmax(dim=1).item())

        next_state, reward, done, _ = env.step(action)
        next_state = np.array(next_state, dtype=np.float32) / 255.0
        buffer.push(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward
        frame_idx += 1

        if len(buffer) >= batch_size:
            states_b, actions_b, rewards_b, next_states_b, dones_b = buffer.sample(batch_size)
            states_v = torch.tensor(states_b, device=device)
            next_states_v = torch.tensor(next_states_b, device=device)
            actions_v = torch.tensor(actions_b, device=device, dtype=torch.int64)
            rewards_v = torch.tensor(rewards_b, device=device)
            done_mask = torch.tensor(dones_b, device=device, dtype=torch.bool)

            q_vals = online_net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

            with torch.no_grad():
                next_actions = online_net(next_states_v).argmax(dim=1)
                next_q = target_net(next_states_v).gather(1, next_actions.unsqueeze(-1)).squeeze(-1)
                next_q = next_q.masked_fill(done_mask, 0.0)
                target_vals = rewards_v + gamma * next_q

            loss = F.mse_loss(q_vals, target_vals)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if frame_idx % sync_target_steps == 0:
            target_net.load_state_dict(online_net.state_dict())

        if done:
            episode += 1
            episode_rewards.append(episode_reward)
            print(f"Episode {episode}, Frame {frame_idx}, Reward {episode_reward:.2f}, Epsilon {eps:.3f}")
            state = env.reset()
            state = np.array(state, dtype=np.float32) / 255.0
            episode_reward = 0

    torch.save(online_net.state_dict(), 'dqn_double.pth')
    env.close()

    # Plot rewards
    plt.figure(figsize=(10,5))
    plt.plot(episode_rewards)
    plt.title('Double DQN Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    train()
