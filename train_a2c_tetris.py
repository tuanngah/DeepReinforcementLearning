import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tetris_env import TetrisEnv
from collections import deque
import time

# ————— CONFIG —————
TOTAL_TIMESTEPS = 200_000
ROLLOUT_LENGTH = 5       
GAMMA = 0.99
LR = 1e-4
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5

# ————— DEVICE —————
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ————— ACTOR-CRITIC NETWORK —————
class ActorCritic(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(ActorCritic, self).__init__()
        c, h, w = input_shape
        # Convolutional backbone
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # Compute conv output size
        def conv2d_size(size, kernel, stride):
            return (size - (kernel - 1) - 1) // stride + 1
        h1 = conv2d_size(h, 8, 4)
        w1 = conv2d_size(w, 8, 4)
        h2 = conv2d_size(h1, 4, 2)
        w2 = conv2d_size(w1, 4, 2)
        h3 = conv2d_size(h2, 3, 1)
        w3 = conv2d_size(w2, 3, 1)
        fc_input_dim = h3 * w3 * 64
        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(fc_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(fc_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = x / 255.0
        features = self.conv(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value.squeeze(-1)

# ————— TRAIN LOOP —————

def train():
    env = TetrisEnv()
    input_shape = env.observation_space.shape
    n_actions = env.action_space.n
    model = ActorCritic(input_shape, n_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    state = env.reset()
    state = np.array(state, dtype=np.float32)
    done = False
    episode_rewards = []
    ep_reward = 0.0
    timestep = 0
    
    while timestep < TOTAL_TIMESTEPS:
        # storage for rollout
        states = []
        actions = []
        rewards = []
        dones = []
        log_probs = []
        values = []
        entropies = []
        
        # collect rollout
        for _ in range(ROLLOUT_LENGTH):
            s_v = torch.tensor([state], device=device)
            logits, value = model(s_v)
            probs = F.softmax(logits, dim=1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            next_state, reward, done, _ = env.step(action.item())
            next_state = np.array(next_state, dtype=np.float32)

            states.append(s_v)
            actions.append(action)
            rewards.append(torch.tensor([reward], device=device))
            dones.append(done)
            log_probs.append(log_prob)
            values.append(value)
            entropies.append(entropy)

            state = next_state
            ep_reward += reward
            timestep += 1

            if done:
                state = env.reset()
                state = np.array(state, dtype=np.float32)
                episode_rewards.append(ep_reward)
                print(f"Episode done, reward={ep_reward}")
                ep_reward = 0.0

            if timestep >= TOTAL_TIMESTEPS:
                break

        # compute last value for bootstrap
        s_v = torch.tensor([state], device=device)
        _, last_value = model(s_v)
        values.append(last_value)

        # compute returns and advantages
        returns = []
        advantages = []
        R = last_value.detach()
        for step in reversed(range(len(rewards))):
            R = rewards[step] + GAMMA * R * (1.0 - dones[step])
            returns.insert(0, R)
            adv = R - values[step]
            advantages.insert(0, adv)

        returns = torch.cat(returns)
        log_probs = torch.cat(log_probs)
        values = torch.cat(values[:-1])
        entropies = torch.cat(entropies)
        actions = torch.cat(actions)

        # compute losses
        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = F.mse_loss(values, returns.detach())
        entropy_loss = -ENTROPY_COEF * entropies.mean()
        loss = policy_loss + VALUE_COEF * value_loss + entropy_loss

        # backprop
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()

    env.close()

    # save model
    torch.save(model.state_dict(), 'a2c_tetris.pth')
    print("Training completed and model saved as a2c_tetris.pth")


if __name__ == '__main__':
    train()
