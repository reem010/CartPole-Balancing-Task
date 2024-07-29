import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
history = []
def run(is_training=True, render=False):

    env = gym.make('CartPole-v1', render_mode='human' if render else None)

    # Divide position, velocity, pole angle, and pole angular velocity into segments
    pos_space = np.linspace(-2.4, 2.4, 10)
    vel_space = np.linspace(-4, 4, 10)
    ang_space = np.linspace(-.2095, .2095, 10)
    ang_vel_space = np.linspace(-4, 4, 10)

    def choose_action(state_p, state_v, state_a, state_av):
        if np.random.uniform(0, 1) < epsilon:
            return env.action_space.sample()
        else:
            return np.argmax(q[state_p, state_v, state_a, state_av, :])


    if(is_training):
        q = np.zeros((len(pos_space)+1, len(vel_space)+1, len(ang_space)+1, len(ang_vel_space)+1, env.action_space.n)) # init a 11x11x11x11x2 array
    else:
        f = open('cartpole.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    learning_rate_a = 0.1 # alpha or learning rate
    discount_factor_g = 0.99 # gamma or discount factor.
    episodes = 100000
    epsilon = 1         # 1 = 100% random actions
    epsilon_decay_rate = 0.00001 # epsilon decay rate
    rng = np.random.default_rng()   # random number generator

    rewards_per_episode = []
    history = []

    i = 0

    # for i in range(episodes):
    for episode in range(episodes):

        state = env.reset()[0]      # Starting position, starting velocity always 0
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)
        state_a = np.digitize(state[2], ang_space)
        state_av = np.digitize(state[3], ang_vel_space)

        action1 = choose_action(state_p, state_v, state_a, state_av)

        terminated = False          # True when reached goal

        total_reward=0

        while(not terminated and total_reward < 10000):

            new_state,reward,terminated,_,_ = env.step(action1)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)
            new_state_a = np.digitize(new_state[2], ang_space)
            new_state_av= np.digitize(new_state[3], ang_vel_space)

            action2 = choose_action(new_state_p, new_state_v, new_state_a, new_state_av)

            if is_training:
                q[state_p, state_v, state_a, state_av, action1] = q[state_p, state_v, state_a, state_av, action1] + learning_rate_a * (
                    reward + discount_factor_g*(q[new_state_p, new_state_v, new_state_a, new_state_av,action2]) - q[state_p, state_v, state_a, state_av, action1]
                )

            state = new_state
            state_p = new_state_p
            state_v = new_state_v
            state_a = new_state_a
            state_av= new_state_av
            action1 = action2

            total_reward+=reward

            if render:
                env.render()

        rewards_per_episode.append(total_reward)

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

    # Save Q table to file
    if is_training:
        f = open('cartpole.pkl','wb')
        pickle.dump(q, f)
        f.close()

    mean_rewards = []
  
    for t in range(i):
        mean_rewards.append(np.mean(rewards_per_episode[max(0, t-100):(t+1)]))
    plt.plot(history)
    plt.savefig(f'cartpoleSarsa.png')
    plt.show()
if __name__ == '__main__':
    run(is_training=True, render=False)
