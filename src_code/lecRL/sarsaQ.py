import gymnasium as gym
import torch
import time
import numpy as np
import random

def print_results(Q, title, grid_shape=(4, 12)):
    """
    Prints the learned policy in a grid format suitable for Cliff Walking.
    """
    # Action mapping for CliffWalking-v0: 0: up, 1: right, 2: down, 3: left
    actions_map = {0: '↑', 1: '→', 2: '↓', 3: '←'}
    
    n_states = Q.shape[0]
    
    # Derive the policy by choosing the action with the highest Q-value for each state
    policy = torch.argmax(Q, dim=1)
    
    # Convert policy numbers to action symbols
    policy_str = [actions_map[p.item()] for p in policy]

    print(f"\n===== {title} result =====")
    print("\nOptimal Policy:")
    # Reshape the policy to the environment's grid dimensions
    print(np.array(policy_str).reshape(grid_shape))


def run_episode(env, Q, render=False):
    """
    Runs a single episode using the learned Q-table to test the policy.
    Always exploits the best-known action (no exploration).
    """
    obs, info = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        if render:
            # Visually render the current state of the environment
            print(f"Current State: {obs}, Q-values: {Q[obs].numpy()}")
            env.render()
            time.sleep(0.25)
            
        # Exploit: Choose the best action from the Q-table
        action = torch.argmax(Q[obs]).item()
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        total_reward += reward
        
    if render:
        # Render the final state (goal or cliff)
        env.render()
        print(f"Episode finished with a total reward of: {total_reward}")
        time.sleep(1)
        
    return total_reward


def q_learning(env, alpha=0.1, gamma=0.99, epsilon=0.1, num_episodes=10000):
    """
    Q-learning algorithm. This is an OFF-POLICY method.
    It learns the value of the optimal policy independently of the agent's actions.
    The update rule uses the maximum Q-value of the next state (max_a' Q(s', a')).
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
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
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # --- Q-Learning Update Rule ---
            # It uses the best possible next action, regardless of which action is actually chosen next.
            best_next_q = torch.max(Q[next_obs])
            Q[obs, action] = Q[obs, action] + alpha * (reward + gamma * best_next_q - Q[obs, action])
            
            obs = next_obs
            
    return Q


def sarsa(env, alpha=0.1, gamma=0.99, epsilon=0.1, num_episodes=10000):
    """
    SARSA algorithm. This is an ON-POLICY method.
    It learns the value of the policy the agent is actually following (including exploration).
    The update rule uses the Q-value of the actual next state-action pair (Q(s', a')).
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    Q = torch.zeros(n_states, n_actions)
    
    for i_episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        
        # Choose the first action based on the current policy
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = torch.argmax(Q[obs]).item()

        while not done:
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # --- SARSA-specific step: Choose the NEXT action from the NEXT state ---
            # This next action is needed for the update rule below.
            if random.uniform(0, 1) < epsilon:
                next_action = env.action_space.sample()  # Explore
            else:
                next_action = torch.argmax(Q[next_obs]).item()  # Exploit

            # --- SARSA Update Rule ---
            # It uses the action that was actually chosen for the next step (next_action).
            # This makes it on-policy, as it learns based on what it does, not what it could have done.
            next_q = Q[next_obs, next_action]
            Q[obs, action] = Q[obs, action] + alpha * (reward + gamma * next_q - Q[obs, action])
            
            # Move to the next state and action for the next iteration
            obs = next_obs
            action = next_action
            
    return Q


if __name__ == '__main__':
    # === 1. Training Phase (no rendering) ===
    # We use render_mode=None for fast training.
    train_env = gym.make('CliffWalking-v1', render_mode=None)
    
    # --- Q-Learning Training ---
    print("Training with Q-Learning...")
    start_time = time.time()
    Q_qlearning = q_learning(train_env, num_episodes=10000)
    end_time = time.time()
    print_results(Q_qlearning, "Q-Learning")
    print(f"Q-Learning Training Time: {end_time - start_time:.4f} seconds\n")

    # --- SARSA Training ---
    print("Training with SARSA...")
    start_time = time.time()
    Q_sarsa = sarsa(train_env, num_episodes=10000)
    end_time = time.time()
    print_results(Q_sarsa, "SARSA")
    print(f"SARSA Training Time: {end_time - start_time:.4f} seconds\n")
    
    train_env.close()

    # === 2. Demonstration Phase (with rendering) ===
    # Create a new environment with render_mode='human' to visualize the results.
    print("\n--- Starting Policy Demonstration ---")
    demo_env = gym.make('CliffWalking-v1', render_mode='human')

    # Demonstrate Q-Learning policy
    print("\n--- Demonstrating Q-Learning Policy (Optimal Path) ---")
    run_episode(demo_env, Q_qlearning, render=True)

    # Demonstrate SARSA policy
    print("\n--- Demonstrating SARSA Policy (Safer Path) ---")
    run_episode(demo_env, Q_sarsa, render=True)

    demo_env.close()