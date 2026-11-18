import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

# Register Atari environments
gym.register_envs(ale_py)

# Create environment
env = gym.make(
    "ALE/MsPacman-v5",
    render_mode="human", #change to 'human' to see render
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
    def __init__(self, nn):
        self.action_nn = DQN(obs.size, env.action_space.n).to(device)
        self.target_nn = DQN(obs.size, env.action_space.n).to(device)
        self.target_nn.load_state_dict(self.action_nn.state_dict())
        self.gamma = 0.9 #discount factor
        self.replay_memory = []
        self.C = 4
        self.memory_minibatch_size = 8
        

    def select_action(self, nn, observation):
        # Convert observation to tensor
        obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        # Get action from the neural network
        action_tensor = nn(obs_tensor)
        value  = max(action_tensor)
        action = torch.argmax(action_tensor).item()
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

        # 3. Convert observations into batched tensors
        states = torch.stack([
            torch.as_tensor(s) for s in states
        ]).unsqueeze(1).float().to(device) / 255.0

        next_states = torch.stack([
            torch.as_tensor(s) for s in next_states
        ]).unsqueeze(1).float().to(device) / 255.0

        # Convert remaining components into tensors
        actions = torch.tensor(
            actions, dtype=torch.long, device=device
        )

        rewards = torch.tensor(
            rewards, dtype=torch.float32, device=device
        )

        terminated = torch.tensor(
            terminated, dtype=torch.float32, device=device
        )

        # 4. Get Q-values for the actions actually taken
        predicted_q = self.action_nn(states)

        predicted_q = predicted_q.gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        # 5. Calculate target Q-values 
        with torch.no_grad():

            next_q = self.target_nn(next_states)

            max_next_q = next_q.max(dim=1).values

            target_q = rewards + self.gamma * (
                1 - terminated
            ) * max_next_q

        loss = self.action_nn.optimize_model(
            predicted_q, target_q
        )

        return loss


    def update_target_nn(self):
        self.target_nn.load_state_dict(self.action_nn.state_dict())
                   

agent = Agent(DQN(obs.size, env.action_space.n).to(device))

#TRAINING LOOP
for step in range(1000):
    # action = env.action_space.sample()
    action = agent.select_action(agent.action_nn, obs)[0]
    

    next_obs, reward, terminated, truncated, info = env.step(action)
    # Store experience in replay memory
    agent.replay_memory.append((obs, action, reward, next_obs, terminated, truncated))

    loss = agent.training_step() #need to show loss per episode?
    if step % 10 == 0:
        print(f"Step: {step}, Current Loss: {loss}")

    obs = next_obs

    if step % agent.C == 0: #every C steps, match target_nn with action_nn
        agent.update_target_nn()

    if terminated or truncated:
        obs, info = env.reset()

env.close()


