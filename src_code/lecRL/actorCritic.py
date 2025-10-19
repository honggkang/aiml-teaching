import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# --- 하이퍼파라미터 ---
learning_rate = 0.0005
gamma = 0.99
num_episodes = 500
report_interval = 100

# --- 공통 신경망 모델 ---
# Actor와 Critic이 신경망의 일부를 공유할 수도 있지만, 여기서는 분리하여 구현
class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.network(x)

class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            # 상태 가치(V(s))는 하나의 스칼라 값이므로 출력은 1
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.network(x)

# --- Actor-Critic agent ---
class ActorCriticAgent:
    def __init__(self, state_size, action_size):
        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

    # Actor-Critic updates every step
    def update_policy(self, state, action, reward, next_state, done):
        state = torch.from_numpy(state).float().unsqueeze(0)
        next_state = torch.from_numpy(next_state).float().unsqueeze(0)
        
        # --- Critic ---
        state_value = self.critic(state)
        next_state_value = self.critic(next_state)

        # TD error (Advantage)
        advantage = reward + gamma * next_state_value * (1 - done) - state_value
        
        # Critic loss function
        critic_loss = advantage.pow(2)
        
        self.critic_optimizer.zero_grad()
        # detach()를 통해 advantage로부터 critic으로 gradient가 흐르지 않도록 함
        critic_loss.backward(retain_graph=True) 
        self.critic_optimizer.step()

        # --- Actor ---
        probs = self.actor(state)
        m = Categorical(probs)
        log_prob = m.log_prob(torch.tensor(action))
        
        # Actor loss function
        # Advantage가 크면(긍정적) 해당 행동의 로그 확률을 더 크게 만들고,
        # 작으면(부정적) 로그 확률을 더 작게 만듦
        actor_loss = -log_prob * advantage.detach()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


def main():
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = ActorCriticAgent(state_size, action_size)
    total_rewards = []

    for i_episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        for t in range(1, 10000):
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            probs = agent.actor(state_tensor)
            m = Categorical(probs)
            action = m.sample().item()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # update policy at each step
            agent.update_policy(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        total_rewards.append(episode_reward)

        if i_episode % report_interval == 0:
            avg_reward = np.mean(total_rewards[-report_interval:])
            print(f'Episode {i_episode}\tAverage Score: {avg_reward:.2f}')

    env.close()

    print("\nTraining finished. Starting trained agent visualization...")

    eval_env = gym.make('CartPole-v1', render_mode='human')
    
    for i in range(10): # 10 episodes
        state, _ = eval_env.reset()
        episode_reward = 0
        # CartPole-v1의 최대 스텝은 500입니다.
        for t in range(500):
            # Use trained policy (actor), no update
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            with torch.no_grad():
                probs = agent.actor(state_tensor)
                m = Categorical(probs)
                action = m.sample().item()
            
            next_state, reward, terminated, truncated, _ = eval_env.step(action)
            
            state = next_state
            episode_reward += reward
            
            if terminated or truncated:
                print(f"Episode {i+1} finished after {t+1} timesteps. Total reward: {episode_reward}")
                break
    
    eval_env.close()    

if __name__ == '__main__':
    main()
