
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

learning_rate = 0.001
gamma = 0.99
num_episodes = 1000
report_interval = 100

# input:observation
# output: action probabilities
class Policy(nn.Module):
    def __init__(self, state_size, action_size):
        super(Policy, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.network(x)

# --- REINFORCE agent ---
class ReinforceAgent:
    def __init__(self, state_size, action_size):
        self.policy = Policy(state_size, action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        self.saved_log_probs = []
        self.rewards = []

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state)
        m = Categorical(probs)
        # sampling
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()

    # Policy update after episode
    def update_policy(self):
        R = 0
        policy_loss = []
        returns = []
        
        # Monte-carlo: get return after episode
        # G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...
        for r in self.rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        
        # G_t normalization
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-6)

        # loss function: log(pi(a|s)) * G_t
        # For Gradient Ascent, add - to the loss
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)

        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

        del self.rewards[:]
        del self.saved_log_probs[:]


def main():
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = ReinforceAgent(state_size, action_size)
    
    total_rewards = []

    for i_episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        # episode (trajectory)
        for t in range(1, 10000):
            action = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            agent.rewards.append(reward)
            episode_reward += reward
            if terminated or truncated:
                break
        
        # update after a episode
        agent.update_policy()
        
        total_rewards.append(episode_reward)

        if i_episode % report_interval == 0:
            avg_reward = np.mean(total_rewards[-report_interval:])
            print(f'Episode {i_episode}\tAverage Score: {avg_reward:.2f}')
    
    env.close()

if __name__ == '__main__':
    main()
