# CartPole-Balancing-Task

## Project Description

This project focuses on training reinforcement learning agents using different algorithms to solve the CartPole-v1 environment from OpenAI's Gym. The goal is to balance a pole on a moving cart for as long as possible by controlling the cart's movements.

## Algorithms Used and Their Mechanisms

### 1. Monte Carlo Control

- **Update Mechanism**: Uses Off-Policy Monte Carlo learning.
- **Update**: Updates the Q-table based on the discounted sum of rewards (G) obtained during an episode.
- **Action Selection**: Epsilon-greedy policy with epsilon decay.
  - **Selection**: Chooses actions either randomly (exploration) or based on Q-values (exploitation).

### 2. Q-Learning

- **Update Mechanism**: Uses the Q-learning algorithm.
- **Update**: Updates the Q-table using the Q-learning formula, incorporating the maximum Q-value of the next state-action pair.
- **Action Selection**: Epsilon-greedy policy with epsilon decay.
  - **Selection**: Randomly samples actions or selects the action with the highest Q-value for exploitation.

### 3. SARSA

- **Update Mechanism**: Uses the SARSA (State-Action-Reward-State-Action) algorithm.
- **Update**: Updates the Q-table based on the current and next state-action pairs and the immediate reward.
- **Action Selection**: Epsilon-greedy policy with epsilon decay.
  - **Selection**: Randomly chooses actions or selects the action with the highest Q-value for exploitation.

## Summary

Each algorithm aims to learn a policy that maximizes the cumulative reward in the CartPole-v1 environment through iterative training episodes. They differ in how they update their Q-tables and select actions, providing insights into the trade-offs between exploration and exploitation in reinforcement learning.

## Dependencies

- `gymnasium` for the CartPole environment.
- `numpy` for numerical operations.
- `matplotlib` for plotting results.
- `pickle` for saving and loading Q-tables.


