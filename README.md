# CartPole-Balancing-Task

## Overview

This project aims to balance a pole on a cart using various reinforcement learning techniques. The primary methods employed are Q-Learning, SARSA, and Monte Carlo methods. The CartPole-v1 environment from OpenAI's Gym is used for training and evaluation.

## Environment

- **CartPole-v1**: A classic control problem where the goal is to balance a pole on a cart by applying appropriate forces to the cart. The environment is rendered to visualize the balancing task.

## Techniques Used

### Q-Learning

- **Initialization**: A Q-table is initialized to zeros, representing expected future rewards for state-action pairs.
- **Action Selection**: Actions are chosen using an epsilon-greedy strategy, balancing exploration and exploitation.
- **Update Rule**: Q-values are updated based on the Bellman equation, incorporating immediate rewards and maximum future rewards.

### Monte Carlo Methods

- **Episode Sampling**: Episodes of interactions with the environment are collected.
- **Return Calculation**: Returns (G) are computed for each state-action pair, considering cumulative discounted rewards.
- **Q-Value Update**: Q-values are updated using the average returns from multiple episodes.

### SARSA (State-Action-Reward-State-Action)

- **Action Selection**: Actions are selected based on the current policy, which includes exploration.
- **Update Rule**: Q-values are updated using the SARSA update rule, accounting for the next action chosen under the current policy.

## Implementation Details

- **State Discretization**: Continuous state spaces (position, velocity, angle, angular velocity) are divided into discrete bins.
- **Epsilon-Greedy Policy**: Epsilon value decreases over time to transition from exploration to exploitation.
- **Training Loop**: Interactions with the environment are performed over a specified number of episodes. Q-values are updated based on collected experiences.
- **Performance Tracking**: Rewards per episode are tracked and plotted to visualize learning progress. The environment is considered solved if the mean rewards exceed a threshold of 195.0.

