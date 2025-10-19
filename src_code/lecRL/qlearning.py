import gymnasium as gym
import torch
import time
import numpy as np
import random


def print_results(Q, title, env):
    """결과를 보기 좋게 출력하는 함수"""
    actions_map = {0: '←', 1: '↓', 2: '→', 3: '↑'}
    
    n_states = env.observation_space.n
    grid_size = int(n_states**0.5)
    
    policy = torch.zeros(n_states, dtype=torch.int)
    for s in range(n_states):
        policy[s] = torch.argmax(Q[s])
    
    policy_str = [actions_map[p.item()] for p in policy]

    print(f"\n===== {title} result =====")
    print("\n Optimal Policy:")
    # Policy in 2D grid format
    print(np.array(policy_str).reshape(grid_size, grid_size))


def run_episode(env, Q, render=False, epsilon=0.0):
    """ Run episode with the learned Q-table and return the result """
    obs, info = env.reset()
    total_reward = 0
    step_idx = 0
    done = False
    
    while not done:
        if render:
            # Visually render current state of the environment
            env.render()
            time.sleep(0.25)
            
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = torch.argmax(Q[obs]).item()  # Exploit
        
        # step() returns 5 values
        obs, reward, terminated, truncated, info = env.step(action)
        # Episode ends when terminated(success/failure) or truncated (out-of-time)
        done = terminated or truncated
        
        total_reward += reward
        step_idx += 1
        
    if render:
        env.render() # render the last state
        
    return total_reward


def q_learning(env, alpha=0.1, gamma=0.99, epsilon=0.1, num_episodes=10000, step_reward=-0.01, hole_reward=-1.0, print_every=1000):
    """
    Q-learning algorithm with modified rewards and Q-value printing.
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    # 1. Initialize Q-table
    Q = torch.zeros(n_states, n_actions)
    
    for i_episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        
        while not done:
            # Epsilon-greedy action selection
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = torch.argmax(Q[obs]).item()  # Exploit
            
            # Take action and observe the outcome
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Modify reward based on the outcome
            if terminated and reward == 0:  # Fell into a hole
                reward = hole_reward
            else:
                reward += step_reward  # Add a small negative reward for each step
            
            # Q-table update
            Q[obs, action] = Q[obs, action] + alpha * (reward + gamma * torch.max(Q[next_obs]) - Q[obs, action])
            
            # Move to the next state
            obs = next_obs
        
        # Print Q-values every 'print_every' episodes
        if (i_episode + 1) % print_every == 0:
            print(f"\nQ-table after episode {i_episode + 1}:")
            print(Q)
            
    return Q


if __name__ == '__main__':
    # is_slippery=False: non-slippery deterministic env
    # is_slippery=True: slippery stochastic env
    env = gym.make('FrozenLake-v1', is_slippery=False, render_mode=None)
    
    # Q-Learning
    start_time = time.time()
    Q_table = q_learning(env, alpha=0.1, gamma=0.99, epsilon=0.1, num_episodes=10000, step_reward=-0.01, hole_reward=-1.0, print_every=1000)
    end_time = time.time()
    print_results(Q_table, "Q-Learning", env)
    print(f"실행 시간: {end_time - start_time:.4f}초")
    
    # Test the learned policy
    print("\n--- Q-Learning Policy Test (3 runs) ---")
    rewards = [run_episode(env, Q_table, epsilon=0.0) for _ in range(3)]
    print(f"Averaged reward: {sum(rewards) / len(rewards)}")
    
    env.close()