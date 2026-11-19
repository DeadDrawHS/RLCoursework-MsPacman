import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque

# Register Atari environments
gym.register_envs(ale_py)

# Create environment
env = gym.make(
    "ALE/MsPacman-v5",
    render_mode=None, # 'human' or None for no rendering
    obs_type="grayscale"
)

obs, info = env.reset()

print("Observation shape:", obs.shape)
print("Observation size:", obs.size)
print("Action space:", env.action_space)

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class DQN(nn.Module):

    def __init__(self, n_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5) # 1 color channel 
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.layer1 = nn.Linear(29008, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
        self.lr = 3e-4
        self.loss = nn.HuberLoss()
        self.optimizer = optim.AdamW(self.parameters(), lr=self.lr, amsgrad=True)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

    def optimize_model(self, y, y2):
        self.optimizer.zero_grad()
        loss = self.loss(y, y2) # list to tensor
        loss.backward()
        self.optimizer.step()
        return loss.item()



class Agent:
    def __init__(self, epsilon,gamma, C, memory_minibatch_size):
        self.action_nn = DQN(env.action_space.n).to(device)
        self.target_nn = DQN(env.action_space.n).to(device)
        self.target_nn.load_state_dict(self.action_nn.state_dict())
        self.gamma = gamma #discount factor
        self.epsilon = epsilon 
        self.replay_memory = deque(maxlen=100000)
        self.C = C
        self.memory_minibatch_size = memory_minibatch_size

    def add_to_replay_memory(self, transition): # Can be made to remove lower TD error transitions if memory is full
        if len(self.replay_memory) >= self.replay_memory.maxlen:
            self.replay_memory.popleft()  # Remove the oldest transition
        self.replay_memory.appendleft(transition)

    def select_action(self, nn, observation):
        obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        # Get action values (vector) from the neural network
        action_tensor = nn(obs_tensor)
        if random.random() < self.epsilon: #Epsilon Greedy Exploration
            value  = max(action_tensor)
            action = torch.argmax(action_tensor).item()
        else:
            action = random.randint(0, env.action_space.n - 1)
            value  = action_tensor[0][action].item()
        
        return (action, value) 

    
    def training_step(self): # LARGELY FROM CHATGPT, NEED TO REWRITE 

        if len(self.replay_memory) < self.memory_minibatch_size:
            return

        batch = random.sample(
            self.replay_memory,
            self.memory_minibatch_size
        )

        # 2. Separate transitions into individual components
        states, actions, rewards, next_states, terminated, truncated = zip(*batch)

        # Convert States and Next states into batch tensors, unsqueeze to add channel dimension 
        states = torch.stack([
            torch.as_tensor(s) for s in states
        ]).unsqueeze(1).float().to(device) / 255.0

        next_states = torch.stack([
            torch.as_tensor(s) for s in next_states
        ]).unsqueeze(1).float().to(device) / 255.0 # Normalise pixel values to [0, 1]  

        # Convert remaining components into tensors
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)

        #Convert Terminated booleans to 0s and 1s for target_q calculation
        terminated = torch.tensor(terminated, dtype=torch.float32).to(device)

        # Takes the q values predicted and index based on the action actually taken. Then remove extra dimension from action 
        predicted_y = self.action_nn(states)
        predicted_y = predicted_y.gather(1, actions).squeeze(1) 

        # predicted_y = torch.tensor([prediction[action] for prediction, action in zip(self.action_nn(states), actions)]).to(device)

        

        # 5. Calculate target Q-values 
        with torch.no_grad():

            next_q = self.target_nn(next_states)

            max_next_q = next_q.max(dim=1).values

            target_y = rewards + self.gamma * (1 - terminated) * max_next_q

        loss = self.action_nn.optimize_model(predicted_y, target_y)

        return loss


    def update_target_nn(self):
        self.target_nn.load_state_dict(self.action_nn.state_dict())


#HYPERPARAMETERS
epsilon = 0.1
gamma = 0.9
C = 4
memory_minibatch_size = 8
episodes = 100 
                   

agent = Agent(epsilon=epsilon, gamma=gamma, C=C, memory_minibatch_size=memory_minibatch_size)

episode = 0
step = 0
episode_reward_sum = 0
#TRAINING LOOP
while episode < episodes:
    # action = env.action_space.sample()
    action = agent.select_action(agent.action_nn, obs)[0]
    

    next_obs, reward, terminated, truncated, info = env.step(action)
    # Store experience in replay memory
    agent.add_to_replay_memory((obs, action, reward, next_obs, terminated, truncated))
    episode_reward_sum += reward

    loss = agent.training_step() #need to show loss per episode?
    if step % 100 == 0:
        print(f"Step: {step}, Current Loss: {loss}")

    obs = next_obs

    if step % agent.C == 0: #every C steps, match target_nn with action_nn
        agent.update_target_nn()

    if terminated or truncated:
        episode += 1
        print(f"Episode: {episode}, Total Reward: {episode_reward_sum} Replay Memory Size: {len(agent.replay_memory)}")
        episode_reward_sum = 0

        obs, info = env.reset()
    step += 1

env.close()


