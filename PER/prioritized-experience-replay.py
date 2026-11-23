import gymnasium as gym
import ale_py
from gymnasium.wrappers import FrameStackObservation, AtariPreprocessing
import torch.nn as nn
import random
import numpy as np
import torch

gym.register_envs(ale_py)

env = gym.make("ALE/MsPacman-v5", render_mode= None, frameskip =1)

# Conversion to grayscale and resizing to 84x84
env = AtariPreprocessing(
    env, 
    screen_size = 84, # originally 210 x 160
    grayscale_obs = True,
    grayscale_newaxis = False,
    scale_obs = False,
    frame_skip = 4, # Atari already has a frame skip of 4, we disable it in the gym.make, so it will follow the frame_skip here
    terminal_on_life_loss = True,
)

# frame stacking
env = FrameStackObservation(env, stack_size=4)

# reset returns a tuple of (initial_observation, info_dictionary, which is used for logging)
# this is used to reset the env, and to give us the initial action to take. 
observation, info = env.reset(seed = 42)

# Q-network Architecture
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        # super is needed here to register the submodules (self.conv and self.fc), because they use items in nn.ModuleList
        super().__init__()
        # adds the layers for a CNN
        self.conv = nn.Sequential(
            #params:(inputchannels, outputchannels, ks, stride)
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size = 3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.conv(x.float() / 255.0)
        x = self.fc(x)
        return x

# 1d array for tree, index 0 = root, left child = 2* i +1 , right child = 2*i+2, i = node index
# parent in 1d array = (i-1) // 2
# Prioritized Experience Replay comes in two parts, sum tree and the PER

# Sum Tree implementation
class SumTree():
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity -1) # the tree is a 1d numpy array
        self.write_ptr = 0
        self.data = [None] * capacity # the buffer that will store our transitions
        self.num_of_entries = 0 # shows how many samples the buffer holds

    def add (self, priority, data): 
        # transition_tuple = np.zeros(self.capacity, dtype=object)
        tree_idx = self.write_ptr + self.capacity -1
        self.update(tree_idx, priority) # From the function below
        # transition_tuple[self.write_ptr] = data
        self.data[self.write_ptr] = data
        self.write_ptr = (self.write_ptr + 1) % self.capacity
        self.num_of_entries = min(self.num_of_entries + 1, self.capacity)

    def update(self, tree_idx, priority):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change
            

    # gets leaf node 
    def get(self, value): 
        # repeat until reach a leaf node
       parent = 0
       
       while parent < self.capacity - 1: # while not a leaf node
            left_child = 2 * parent + 1
            right_child = left_child + 1
            if value <= self.tree[left_child]:
                parent = left_child
            else:
                value -= self.tree[left_child]
                parent = right_child

       data_idx = parent - (self.capacity -1) # opposite of the add function, to get index of the data in the buffer 
       return parent, self.tree[parent], self.data[data_idx] # parent here is the leaf node index

# PER implementation
class PrioritizedReplayBuffer():
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment_per_sample=0.001, epsilon=1e-6):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sample = beta_increment_per_sample
        self.epsilon = epsilon
        self.max_priority = 1.0 

    def add(self, state, action, reward, next_state, done):
        self.transition = (state, action, reward, next_state, done)
        priority = self.max_priority # New entries are added with max priority so that they arent left out of the sampling
        self.tree.add(priority, self.transition)

    
    def sample(self, batch_size):
        # stratified sampling
        total = self.tree.tree[0]
        segment = total / batch_size 
        leaf_array = []
        priority_array = []
        data_array = []
        
        for i in range(batch_size):
            value = random.uniform(i * segment, (i+1) * segment)
            leaf_idx, priority, data = self.tree.get(value)
            leaf_array.append(leaf_idx)
            priority_array.append(priority)
            data_array.append(data)
        
        priorities = np.array(priority_array)
        probability = priorities / total
        weights = (1 / (self.tree.num_of_entries * probability)) ** self.beta 

        normalized_weights = weights / np.max(weights)

        self.beta = min(1.0, self.beta + self.beta_increment_per_sample)

        return(data_array, leaf_array, normalized_weights)

    def update_priorities(self, indices, td_errors):
        for (idx, error) in zip(indices, td_errors):
            p = (abs(error) + self.epsilon) ** self.alpha

            self.tree.update(idx, p)
            self.max_priority = max(self.max_priority, p)

class PERAgent():
    def __init__(self, online_net, target_net, buffer_capacity):
        self.online_net = online_net
        self.target_net = target_net
        target_net.load_state_dict(online_net.state_dict())
        target_net.eval()

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = "cpu"

        self.online_net.to(self.device)
        self.target_net.to(self.device)

        self.optimizer = torch.optim.Adam(online_net.parameters(), lr=1e-4)
        self.loss_function = torch.nn.HuberLoss(delta=1.0, reduction = 'none')
        self.buffer = PrioritizedReplayBuffer(capacity = buffer_capacity) # I think this is supposed to hae more parameters, 
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay_steps = 2000000
        self.step_count = 0


    def select_action(self, state):

        current_epsilon = max((self.epsilon_end),(self.epsilon_start - (self.step_count / self.epsilon_decay_steps)))

        if random.random() < current_epsilon:
            action = env.action_space.sample()
            return action
        else:
            state_tensor = torch.tensor(state).unsqueeze(0).to(self.device) 
            with torch.no_grad():
                q_values = self.online_net(state_tensor) 
                return q_values.argmax().item()


    def train_step(self, batch_size):
        # Check if buffer has enough samples:
        if self.buffer.tree.num_of_entries < batch_size:
            return # not enough data yet

        data_array, leaf_indices, weights = self.buffer.sample(batch_size)

        # Extract all the data out from the transition tuple
        states, actions, rewards, next_states, dones = zip(*data_array)
        
        # Conversion to tensor, datatypes and to chose the device to run the training step on
        # Converstion to batched python tensors

        """
        Expected shapes
        states:      [batch_size, 4, 84, 84]
        actions:     [batch_size, 1]
        rewards:     [batch_size, 1]
        next_states: [batch_size, 4, 84, 84]
        dones:       [batch_size, 1]
        weights:     [batch_size, 1]
        unsqueeze to add the dimentionality at the end (turns into column tensor), shape compatible operations
        """

        states = torch.tensor(np.array(states), dtype = torch.float32, device = self.device)
        actions = torch.tensor(np.array(actions), dtype = torch.long, device = self.device).unsqueeze(1)
        rewards = torch.tensor(np.array(rewards), dtype = torch.float32, device = self.device).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype = torch.float32, device = self.device)
        dones = torch.tensor(np.array(dones), dtype= torch.float32, device = self.device).unsqueeze(1)
        weights = torch.tensor(weights, dtype = torch.float32, device = self.device).unsqueeze(1)

        current_q = self.online_net(states).gather(1, actions)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            target_q = rewards + (1 - dones) * 0.99 * max_next_q

        loss_elements = self.loss_function(current_q, target_q)
        loss = (weights * loss_elements).mean()

        # backprop                   
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Compute TD errors 
        td_error = (target_q - current_q).abs().detach().cpu().numpy().flatten()
        self.buffer.update_priorities(leaf_indices, td_error)

# Main training loop
# Instantiate agents and networks

num_actions = env.action_space.n
online_net = DQN(input_shape = (4, 84, 84), num_actions=num_actions)
target_net = DQN(input_shape=(4, 84, 84), num_actions=num_actions)
agent = PERAgent(online_net, target_net, buffer_capacity=100000) # tweak this buffer_capacity

# Define training parameters
WARMUP_STEPS = 5000
BATCH_SIZE = 32
TRAIN_FREQ = 4
TARGET_UPDATE_FREQ = 2000
MAX_EPISODES = 1000

# Replay Buffer Warmup Loop
state, _ = env.reset()

for step in range(WARMUP_STEPS):
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    agent.buffer.add(state, action, reward, next_state, done)
    state = next_state
    if done:
        state, _ = env.reset()

total_steps = 0
for episode in range(1, MAX_EPISODES + 1):
    state, _ = env.reset()
    episode_reward = 0
    done = False

    while not done:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.buffer.add(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        total_steps += 1
        agent.step_count += 1

        if total_steps % TRAIN_FREQ == 0:
            agent.train_step(BATCH_SIZE)

        if total_steps % TARGET_UPDATE_FREQ == 0:
            agent.target_net.load_state_dict(agent.online_net.state_dict())


    # straight up AIed this logging message 
    # Calculate current epsilon for logging
    current_eps = max(agent.epsilon_end, agent.epsilon_start - (agent.step_count / agent.epsilon_decay_steps))
    print(f"Episode {episode:4d}/{MAX_EPISODES} | Steps: {total_steps:7d} | Score: {episode_reward:6.1f} | Epsilon: {current_eps:.3f} | Buffer: {agent.buffer.tree.num_of_entries}")



""" TESTS """

# Tree test 
"""
tree = SumTree(capacity = 4)
tree.add(1, "state_A")
tree.add(2, "state_B")
tree.add(3, "state_C")
tree.add(4, "state_D")
print("Total Priority: ", tree.tree[0])
idx, priority, data = tree.get(3)
print("Sampled: ", data)
"""

# small test to check the shape of the output of the network
"""
network = DQN(input_shape=(4, 84, 84),
              num_actions=env.action_space.n
              )
dummy_input = torch.zeros(1,4,84,84) # zeros creates tensor
output = network(dummy_input)
print("Output shape: ", output.shape)
"""