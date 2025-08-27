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


def value_iteration(env, gamma=0.99, theta=1e-8):
    """
    Value iteration algorithm.
    After finding optimal value V*, derive optimal policy pi* from it.
    done: s=terminal state(16)
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    # 1. Value initializatio to 0
    V = torch.zeros(n_states)
    
    # Access transition probability model (P) of the original environment
    unwrapped_env = env.unwrapped
    
    while True:
        delta = 0
        # 2. Iteration over all states
        for s in range(n_states): # s = row *grid_size + col
            v_old = V[s].clone()
            
            # Calculate Q-value over all possible actions in current state (s)
            q_values = torch.zeros(n_actions)
            for a in range(n_actions):
                # unwrapped_env.P[s][a] returns probability, next_state, reward, done
                for prob, next_state, reward, done in unwrapped_env.P[s][a]:
                    q_values[a] += prob * (reward + gamma * V[next_state])
            
            # 3. State value update by Bellman optimality equation
            # V(s) = max_a Q(s, a)
            V[s] = torch.max(q_values)
            
            # to check convergence, store the largest delta value
            delta = max(delta, torch.abs(v_old - V[s]))
            
        # 4. termination when convergence is achieved
        if delta < theta:
            break
            
    # 5. Get optimal policy form the optimal state value
    policy = torch.zeros(n_states, dtype=torch.int)
    for s in range(n_states):
        q_values = torch.zeros(n_actions)
        for a in range(n_actions):
            for prob, next_state, reward, done in unwrapped_env.P[s][a]:
                q_values[a] += prob * (reward + gamma * V[next_state])
        
        # Select the action with the highest Q-value
        policy[s] = torch.argmax(q_values)
        
    return V, policy


if __name__ == '__main__':
    # is_slippery=False: non-slippery deterministic env
    # is_slippery=True: slippery stochastic env
    env = gym.make('FrozenLake-v1', is_slippery=False, render_mode='human')
    
    # --- Value Iteration ---
    start_time = time.time()
    V_vi, policy_vi = value_iteration(env, gamma=0.99)
    end_time = time.time()
    print_results(V_vi, policy_vi, "Value Iteration")
    print(f"Executed time: {end_time - start_time:.4f}sec")
    
    # --- Learned policy test ---
    print("\n--- Value Iteration Policy Test (3 runs) ---")
    rewards = [run_episode(env, policy_vi) for _ in range(3)]
    print(f"Averaged reward: {sum(rewards) / len(rewards)}")
   
    env.close()
