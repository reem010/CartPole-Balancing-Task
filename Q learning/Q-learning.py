import gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
history = []
random.seed(77777)
def run_q_learning(episodes=1000, render=False):
    env = gym.make('CartPole-v1', render_mode="human" if render else None)
    
    # Define state space discretization
    pos_space = np.linspace(-2.4, 2.4, 10)
    vel_space = np.linspace(-4, 4, 10)
    ang_space = np.linspace(-0.2095, 0.2095, 10)
    ang_vel_space = np.linspace(-4, 4, 10)

    # Initialize Q-table
    q_table_shape = (len(pos_space) + 1, len(vel_space) + 1, len(ang_space) + 1, len(ang_vel_space) + 1, env.action_space.n)
    q_table = np.zeros(q_table_shape)

    learning_rate = 0.1  # Alpha
    discount_factor = 0.99  # Gamma
    epsilon = 1.0  # Initial epsilon for epsilon-greedy policy
    epsilon_decay_rate = 0.001  # Epsilon decay rate
    rewards_per_episode = []

    for episode in range(episodes):
        state = env.reset()[0]
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)
        state_a = np.digitize(state[2], ang_space)
        state_av = np.digitize(state[3], ang_vel_space)

        total_reward = 0
        terminated = False
        
        while not terminated:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Random action
            else:
                action = np.argmax(q_table[state_p, state_v, state_a, state_av, :])  # Greedy action

            new_state, reward, terminated, _, _2 = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)
            new_state_a = np.digitize(new_state[2], ang_space)
            new_state_av = np.digitize(new_state[3], ang_vel_space)

            # Update Q-value using Q-learning equation
            q_table[state_p, state_v, state_a, state_av, action] += learning_rate * (
                reward + discount_factor * np.max(q_table[new_state_p, new_state_v, new_state_a, new_state_av, :]) - q_table[state_p, state_v, state_a, state_av, action]
            )

            state_p, state_v, state_a, state_av = new_state_p, new_state_v, new_state_a, new_state_av
            total_reward += reward

            if render:
                env.render()
        rewards_per_episode.append(total_reward)
        
        if render:
          print(rewards_per_episode[max(0, len(rewards_per_episode) - 4) : -1], len(rewards_per_episode))
        
        # Decay epsilon
        epsilon = max(epsilon - epsilon_decay_rate, 0.1)
        
        # Print progress
        if (episode + 1) % 100 == 0:
            mean_rewards = np.mean(rewards_per_episode[-100:])
            
            print(f"Episode {episode + 1}/{episodes}, Mean Rewards (last 100): {mean_rewards:.2f}")
            history.append(mean_rewards)
            rewards_per_episode = []

            # Check if environment is solved
            if mean_rewards >= 195.0:
                print(f"Environment solved in {episode + 1} episodes!")
                break

    env.close()

    # Save Q-table to file
    with open('cartpole_q_table.pkl', 'wb') as f:
        pickle.dump(q_table, f)

    # Plot rewards per episode
    plt.plot(history)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Q-Learning Training for CartPole-v1')
    plt.show()

if __name__ == '__main__':
    run_q_learning(episodes=50000, render=False)
