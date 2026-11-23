import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import ale_py
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
import numpy as np

gym.register_envs(ale_py)

"""
Standard DQN gives out like discrete answers, which leaves out like a lot of additional information

Formally, it ignores risk and variance within the environment
Distributional RL gives us a probability distribution of possible scores if I take an action

Mechanism:
To compute this for nns, we create a fixed histogram of 41 discrete values between a min and max return value
(Usually -10 - 10), so we will have 51 intervals between -10 and 10. These are called support atoms

1. To pick an action, calculate expected value by taking the weighted sum of the state-action Q-table, then take
    argmax
2. Bellman shift. Take step and get reward, shift return atom 
3. the shift will cause our atom to land between the fixed interval values, so probability is split, take the neighbour
    with the higher probability
4. Use Cross-Entropy loss to train the network 

This improves DQN through allowing the agents conv nn to learn much richer representations of the game state 
(more info)

"""

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Stolen from PER
env = gym.make("ALE/MsPacman-v5", render_mode=None, frameskip=1)
# Conversion to grayscale and resizing to 84x84
env = AtariPreprocessing(
    env, 
    screen_size=84,
    grayscale_obs=True,
    grayscale_newaxis=False,
    scale_obs=False,
    frame_skip=4,
    terminal_on_life_loss=True,
)
# Frame stacking
env = FrameStackObservation(env, stack_size=4)

class C51Network(nn.Module):
    def __init__(self, num_actions, num_atoms = 51):
        super().__init__()
        self.v_min = -10.0
        self.v_max = 10.0
        self.num_actions = num_actions
        self.num_atoms = num_atoms


        # Creating the range of bins for the atoms, linspace creates a 1d tensor, which will hold our values, params = (min num, max num, number of intervals)
        supports =  torch.linspace(self.v_min, self.v_max, self.num_atoms)

        self.register_buffer("supports", torch.linspace(self.v_min, self.v_max, self.num_atoms))

        self.conv = nn.Sequential(
            # we want dimensions 3136 = 64 x 7 x 7 
            nn.Conv2d(4, 32, kernel_size = 8, stride = 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size = 4, stride = 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size = 3, stride = 1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.linear = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions * num_atoms)

        )

    def get_q_values(self, x):
        probs =self.forward(x)
        # shape: ( batch, num_actions, 51)
        # Mulitply the probabilities by the actual atom value and sum the last dimension(dim=-1)  
        q_values = (probs * self.supports).sum(dim=-1)
        return q_values


    def forward(self, x):
        x = x.float() / 255.0
        x = self.conv(x)
        output = self.linear(x)

        output = output.view(-1, self.num_actions, self.num_atoms) # reshape
        output = F.softmax(output, dim=-1) # get a probability distribution 
        return output

    # Bellman Update for categorical projection
    """

    """

    # Categorical Projection (Vectorized PyTorch implementation for massive speedup)
    @staticmethod
    def project_q_value_distributions(next_dist, rewards, dones, gamma, num_atoms, v_min, v_max, supports):
        batch_size = rewards.size(0)
        spacing_between_bins = (v_max - v_min) / (num_atoms - 1)

        # 1. Vectorized Bellman Shift:
        # Instead of looping over 51 atoms one by one, compute the shift for all (batch_size, num_atoms) simultaneously.
        # supports.unsqueeze(0) expands shape to (1, num_atoms). PyTorch broadcasts (batch_size, 1) + (1, num_atoms).
        Tz = torch.clamp(rewards + (1 - dones) * gamma * supports.unsqueeze(0), v_min, v_max)

        # 2. Continuous atom indices across the entire batch: shape (batch_size, num_atoms)
        b = (Tz - v_min) / spacing_between_bins
        lower_bound = b.floor().long()
        upper_bound = b.ceil().long()

        # Handle edge case where b lands exactly on an integer (lower_bound == upper_bound).
        # Nudge bounds so the probability mass weights (upper - b) and (b - lower) don't both collapse to 0.
        lower_bound[(upper_bound > 0) * (lower_bound == upper_bound)] -= 1
        upper_bound[(lower_bound < (num_atoms - 1)) * (lower_bound == upper_bound)] += 1

        # 3. Compute probability mass fractions for all atoms and batch items simultaneously
        mass_lower = next_dist * (upper_bound.float() - b)
        mass_upper = next_dist * (b - lower_bound.float())

        # 4. Vectorized accumulation using 1D index_add_ (replaces the nested 'for j' and 'for i' Python loops):
        # We calculate an offset per batch item so we can flatten proj_dist into 1D and write directly in C++.
        proj_dist = torch.zeros((batch_size, num_atoms), device=rewards.device)
        offset = (
            torch.linspace(0, (batch_size - 1) * num_atoms, batch_size, device=rewards.device)
            .long()
            .unsqueeze(1)
            .expand(batch_size, num_atoms)
        )

        # Flattened accumulation in C++:
        proj_dist.view(-1).index_add_(0, (lower_bound + offset).view(-1), mass_lower.view(-1))
        proj_dist.view(-1).index_add_(0, (upper_bound + offset).view(-1), mass_upper.view(-1))

        return proj_dist


class ReplayBuffer:

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def add(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


GAMMA = 0.99
BATCH_SIZE = 32


class C51Agent():
    def __init__(self, num_actions):
        self.device = device
        self.online_net = C51Network(num_actions).to(device)
        self.target_net = C51Network(num_actions).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=1e-4)
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99995
        self.num_actions = num_actions
        self.gamma = GAMMA
        self.batch_size = BATCH_SIZE
        

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        else:
            with torch.no_grad(): # during gameplay we dont want gradients
            
                state_tensor = torch.from_numpy(np.array(state)).unsqueeze(0).to(self.device)
                q_values = self.online_net.get_q_values(state_tensor)

            return q_values.argmax(dim=1).item()


    def learn(self, replay_buffer):
        if len(replay_buffer) < self.batch_size:
            return

        experiences = replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.tensor(np.array(states), dtype = torch.float32, device = self.device)
        actions = torch.tensor(np.array(actions), dtype = torch.long, device = self.device).unsqueeze(1) # .gather needs int64 or long
        rewards = torch.tensor(np.array(rewards), dtype = torch.float32, device = self.device).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype = torch.float32, device = self.device)
        dones = torch.tensor(np.array(dones), dtype = torch.float32, device = self.device).unsqueeze(1)


        with torch.no_grad():
            next_q = self.target_net.get_q_values(next_states)
            next_actions = next_q.argmax(dim=1, keepdim=True)

            next_dist_all = self.target_net(next_states)

            next_actions_expanded = next_actions.unsqueeze(-1).expand(-1, -1, self.online_net.num_atoms)
            next_dist = next_dist_all.gather(1,next_actions_expanded).squeeze(1) 

            target_dist = self.online_net.project_q_value_distributions(next_dist, rewards, dones, self.gamma, self.online_net.num_atoms, self.online_net.v_min, self.online_net.v_max, self.online_net.supports)

        current_dist_all = self.online_net(states)

        actions_expanded = actions.unsqueeze(-1).expand(-1, -1, self.online_net.num_atoms)
        pred_dist = current_dist_all.gather(1, actions_expanded).squeeze(1)

        loss = - (target_dist * torch.log(pred_dist + 1e-8)).sum(dim=-1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()



if __name__ == "__main__":
    # time to start the training loop 

    num_actions = env.action_space.n
    agent = C51Agent(num_actions)
    replay_buffer = ReplayBuffer(capacity=100000)

    WARMUP_STEPS = 5000
    TARGET_UPDATE_FREQ = 2000
    TRAIN_FREQ = 4
    MAX_EPISODES = 500
    total_steps = 0

    for episode in range(MAX_EPISODES):

        state, info = env.reset()
        done = False    
        episode_reward = 0

        while not done:
            total_steps += 1
            # warmup vs action selection
            if total_steps < WARMUP_STEPS:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)

            # step environment
            next_state, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated

            episode_reward += reward

            # save to buffer:
            replay_buffer.add((state, action, reward, next_state, done))
            state = next_state

            # Train every TRAIN_FREQ steps after warmup
            if total_steps >= WARMUP_STEPS and total_steps % TRAIN_FREQ == 0:
                agent.learn(replay_buffer)

            # Update Target Network with the same weights as of the online network periodically
            if total_steps >= WARMUP_STEPS and total_steps % TARGET_UPDATE_FREQ == 0:
                agent.target_net.load_state_dict(agent.online_net.state_dict())

            # the decay epsilon
            if total_steps >= WARMUP_STEPS:
                agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        print(f"Episode {episode} | Reward: {episode_reward:.2f} | Epsilon: {agent.epsilon:.3f} | Total Steps: {total_steps}")







        




                