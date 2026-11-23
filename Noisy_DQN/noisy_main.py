'''Run from the repo root (with the venv's Python):
.venv\\Scripts\\python NoisyDQN/noisy_main.py
'''

import math
import os
import sys
from collections import deque

import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the shared pieces from the vanilla DQN instead of copying them
HERE = os.path.dirname(os.path.abspath(__file__))                   # .../NoisyDQN
sys.path.insert(0, os.path.join(HERE, "..", "DQN"))                 # so "import main" finds DQN/main.py

from main import Agent, preprocess_observation                      # DQN/main.py (safe: its training only runs under __main__)
from customEnvironment import PacmanRewardWrapper                   # DQN/customEnvironment.py

gym.register_envs(ale_py)


class NoisyLinear(nn.Module):

    def __init__(self, in_features, out_features, sigma_0=0.5): # following the original paper [1]
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_0 = sigma_0

        # Learnable means: play the role of nn.Linear's normal weight/bias
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))

        # Learnable noise scales: how much randomness each weight gets
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Noise samples: register_buffer = moves with .to(device) and is saved in state_dict, but is NOT a parameter, so the optimiser never touches it
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.in_features)                    # paper's init: mu ~ U[-1/sqrt(p), 1/sqrt(p)]
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.sigma_0 / math.sqrt(self.in_features))   # sigma = sigma_0 / sqrt(p)
        self.bias_sigma.data.fill_(self.sigma_0 / math.sqrt(self.in_features))

    def _scaled_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)          # standard normal samples
        return x.sign() * x.abs().sqrt()                             # f(x) = sign(x) * sqrt(|x|), from the paper

    def reset_noise(self):
        eps_in = self._scaled_noise(self.in_features)                # one noise vector per input dimension, one per output dimension, then outer product to get a full noise matrix
        eps_out = self._scaled_noise(self.out_features)
        self.weight_epsilon.copy_(torch.outer(eps_out, eps_in))      # (out x in) matrix from two vectors -> cheap
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x):
        if self.training:                                            # .train() mode -> noisy weights (explore)
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:                                                        # .eval() mode -> mean weights only (no noise),
            weight = self.weight_mu                                  #   use this for clean evaluation runs
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class NoisyDQN(nn.Module):

    def __init__(self, action_size):
        super().__init__()

        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)      #   (paper only makes the dense layers noisy)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = NoisyLinear(64 * 7 * 7, 512)
        self.fc2 = NoisyLinear(512, action_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def reset_noise(self):
        self.fc1.reset_noise()
        self.fc2.reset_noise()

    def mean_sigma(self):                                            # monitoring purposes purely: average |sigma| across all noise
        sigmas = [self.fc1.weight_sigma, self.fc2.weight_sigma]      # sigma weights
        return torch.cat([s.abs().flatten() for s in sigmas]).mean().item()  # the net "turning down" its own exploration

class NoisyAgent(Agent):

    def __init__(self, action_size):
        super().__init__(action_size)

        self.q_network = NoisyDQN(action_size).to(self.device)       # noisy online network
        self.target_network = NoisyDQN(action_size).to(self.device)  # noisy target network
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.target_network.train()
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.0001)  # rebuild: the old optimiser. Points at the old DQN's parameters. Same lr as DQN = fair test.

    def choose_action(self, state):
        # Noisy DQN ALWAYS takes the argmax - the noise in fc1/fc2 provides the exploration.
        with torch.no_grad():
            state = state.to(self.device, dtype=torch.float32) / 255.0
            q_values = self.q_network(state)
            return torch.argmax(q_values, dim=1).item()

    def count_step(self):
        # nothing to decay. Still count steps so environment_steps stays useful for logging.
        self.environment_steps += 1

    def learn(self):
        super().learn()
        self.q_network.reset_noise()                                 # fresh noise for the next action + next update
        self.target_network.reset_noise()                            # independent noise for the target network

def main():
    env = gym.make(
        "ALE/MsPacman-v5",
        render_mode="rgb_array",
        mode=2,
        difficulty=0
    )

    env = gym.wrappers.RecordVideo(
        env,
        video_folder=os.path.join(HERE, "Evidence", "Videos"),
        episode_trigger=lambda episode_id: episode_id % 10 == 0,
        name_prefix="noisy-pacman-video"                             # prefix so videos aren't confused with DQN's
    )

    env = PacmanRewardWrapper(env, -20, -0.01, 3)                    # SAME reward shaping as DQN

    agent = NoisyAgent(env.action_space.n)
    episodes = 100
    rewards = []
    sigmas = []                                                      # track exploration level over time

    for episode in range(episodes):
        observation, info = env.reset()
        total_reward = 0
        done = False
        frames = deque(maxlen=4)
        first_frame = preprocess_observation(observation)
        for _ in range(4):
            frames.append(first_frame)

        while not done:
            state = torch.stack(tuple(frames))
            state = state.unsqueeze(0)
            action = agent.choose_action(state)                      # this is now always greedy

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            frames.append(preprocess_observation(next_observation))
            next_state = torch.stack(tuple(frames))
            agent.remember(state.squeeze(0), action, reward, next_state, done)

            agent.learn()                                            # this resamples the noise

            total_reward += reward
            agent.count_step()                                   # this only counts steps

        rewards.append(total_reward)
        sigmas.append(agent.q_network.mean_sigma())
        print(f"Episode {episode}: {total_reward:.2f}")
        print(f"Mean |sigma|: {sigmas[-1]:.5f}  (steps: {agent.environment_steps})")  # sigma replaces epsilon

    env.close()
    return rewards, sigmas


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rewards, sigmas = main()

    plots_dir = os.path.join(HERE, "Evidence", "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 7))
    ax1.plot(rewards)
    ax1.set_ylabel("Total Reward")
    ax1.set_title("Noisy DQN Training Progress")
    ax2.plot(sigmas)                                                 # second panel - exploration noise over time
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Mean |sigma|")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "noisy_training_progress.png"))
    plt.show()


# Reference:
# [1] https://arxiv.org/abs/1706.10295