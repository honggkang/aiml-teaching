import gymnasium as gym
import torch
import time
import numpy as np


def print_results(V, policy, title):
    # policy (0: Left, 1: Down, 2: Right, 3: Up)
    actions_map = {0: '←', 1: '↓', 2: '→', 3: '↑'}
    policy_str = [actions_map[p.item()] for p in policy]
    
    grid_size = int(V.shape[0]**0.5)
    
    print(f"\n===== {title} result =====")
    print("Optimal Value Function:")
    # State value in 2D grid format
    print(V.reshape(grid_size, grid_size).numpy())
    
    print("\n Optimal Policy:")
    # Policy in 2D grid format
    print(np.array(policy_str).reshape(grid_size, grid_size))


def run_episode(env, policy, render=False):
    """ Run episode with the learned policy and return the result """
    obs, info = env.reset()
    total_reward = 0
    step_idx = 0
    done = False
    
    while not done:
        if render:
            # Visually render current state of the environment
            env.render()
            time.sleep(0.25)
            
        # Select action by policy
        action = policy[obs].item()
        # step() returns 5 values
        obs, reward, terminated, truncated, info = env.step(action)
        # Episode ends when terminated(success/failure) or truncated (out-of-time)
        done = terminated or truncated
        
        total_reward += reward
        step_idx += 1
        
    if render:
        env.render() # render the last state
        
    return total_reward


def policy_iteration(env, gamma=0.99, theta=1e-8):
    """
    Policy Iteration
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    unwrapped_env = env.unwrapped
    
    # 1. Initialize a random policy
    policy = torch.randint(0, n_actions, (n_states,))
    
    while True:
        # --- 2. Policy Evaluation ---
        # Evaluate the value function for the current policy
        V = torch.zeros(n_states)
        while True:
            delta = 0
            for s in range(n_states):
                v_old = V[s].clone()
                
                action = policy[s].item()
                
                # Bellman expectation equation to calculate value function
                new_v = 0
                for prob, next_state, reward, done in unwrapped_env.P[s][action]:
                    new_v += prob * (reward + gamma * V[next_state])
                V[s] = new_v
                
                delta = max(delta, torch.abs(v_old - V[s]))
            
            # Value function converged
            if delta < theta:
                break
        
        # --- 3. Policy Improvement ---
        # Find a better policy pi' using the evaluated value function V_pi
        policy_stable = True
        for s in range(n_states):
            action_old = policy[s].clone()
            
            # Q-value calculation for all actions in state s
            q_values = torch.zeros(n_actions)
            for a in range(n_actions):
                for prob, next_state, reward, done in unwrapped_env.P[s][a]:
                    q_values[a] += prob * (reward + gamma * V[next_state])
            
            # greedy update
            policy[s] = torch.argmax(q_values)
            
            # Check if policy changed
            if action_old != policy[s]:
                policy_stable = False
                
        # 4. Convergence check
        if policy_stable:
            break
            
    return V, policy


if __name__ == '__main__':
    # is_slippery=False: non-slippery deterministic env
    # is_slippery=True: slippery stochastic env
    env = gym.make('FrozenLake-v1', is_slippery=False, render_mode='human')
    
    obs, info = env.reset()
    env.render()
    time.sleep(2)

    # --- Policy Iteration ---
    start_time = time.time()
    V_pi, policy_pi = policy_iteration(env)
    end_time = time.time()
    print_results(V_pi, policy_pi, "Policy Iteration")
    print(f"Executed time: {end_time - start_time:.4f}sec")

    # --- Learned policy test ---
    print("\n--- Policy Iteration Policy Test (3 runs) ---")
    rewards = [run_episode(env, policy_pi) for _ in range(3)]
    print(f"Averaged reward: {sum(rewards) / len(rewards)}")

    env.close()
