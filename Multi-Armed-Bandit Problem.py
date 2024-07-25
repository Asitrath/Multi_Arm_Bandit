import numpy as np
import matplotlib.pyplot as plt
import random


class GaussianBandit:
    def __init__(self):
        self._arm_means = np.random.uniform(0., 1., 10)  # Sample some means
        self.n_arms = len(self._arm_means)
        self.rewards = []
        self.total_played = 0

    def reset(self):
        self.rewards = []
        self.total_played = 0

    def play_arm(self, a):
        reward = np.random.normal(self._arm_means[a], 1.)  # Use sampled mean and covariance of 1.
        self.total_played += 1
        self.rewards.append(reward)
        return reward


def greedy(bandit, timesteps):
    rewards = np.zeros(bandit.n_arms)
    n_plays = np.zeros(bandit.n_arms)
    Q = np.zeros(bandit.n_arms)
    possible_arms = range(bandit.n_arms)

    for x in range(bandit.n_arms):
        n_plays[x] = 1
        rewards[x] = bandit.play_arm(x)
        Q[x] = rewards[x]/n_plays[x]
        
    
    # Main loop
    while bandit.total_played < timesteps:
        # This example shows how to play a random arm:
        # a = random.choice(possible_arms)
        # reward_for_a = bandit.play_arm(a)

        max_index = np.argmax(Q) 
        reward_for_a = bandit.play_arm(max_index)

        
        #Logic ---> Q(n+1) = Q(n) + [1/n(rewards(n)-Q(n))]
        Q[max_index] = Q[max_index] + (1/(n_plays[max_index]))*(rewards[max_index] - Q[max_index])
        rewards[max_index] = reward_for_a
        n_plays[max_index] += 1
    
    
def epsilon_greedy(bandit, timesteps):

    #reward_for_a = bandit.play_arm(0)  # Just play arm 0 as placeholder
    rewards = np.zeros(bandit.n_arms)
    n_plays = np.zeros(bandit.n_arms)
    Q = np.zeros(bandit.n_arms)
    possible_arms = range(bandit.n_arms)
    
    # init variables (rewards, n_plays, Q) by playing each arm once
    for x in range(bandit.n_arms):
        n_plays[x] = 1
        rewards[x] = bandit.play_arm(x)
        Q[x] = rewards[x]/n_plays[x]
    
    epsilon = 0.1
    
    while bandit.total_played < timesteps:
        random_val = np.random.random(1)[0]
        if random_val < epsilon: #if random value less than probability epsilon then choose random action
            #choose arm randomly    
            random_arm = random.choice(possible_arms)
            reward_for_a = bandit.play_arm(random_arm)
            Q[random_arm] = Q[random_arm] + (1/(n_plays[random_arm]))*(rewards[random_arm] - Q[random_arm])
            rewards[random_arm] = reward_for_a
            n_plays[random_arm] += 1

        else: # choose greedy action
            max_index = np.argmax(Q)
            reward_for_a = bandit.play_arm(max_index)
            Q[max_index] = Q[max_index] + (1/(n_plays[max_index]))*(rewards[max_index] - Q[max_index])
            rewards[max_index] = reward_for_a
            n_plays[max_index] += 1

def main():
    n_episodes = 10000 # TODO: set to 10000 to decrease noise in plot
    n_timesteps = 1000
    rewards_greedy = np.zeros(n_timesteps)
    rewards_egreedy = np.zeros(n_timesteps)

    for i in range(n_episodes):
        if i % 100 == 0:
            print("current episode: " + str(i))

        b = GaussianBandit()  # initializes a random bandit
        greedy(b, n_timesteps)
        rewards_greedy += b.rewards

        b.reset()  # reset the bandit before running epsilon_greedy
        epsilon_greedy(b, n_timesteps)
        rewards_egreedy += b.rewards

    rewards_greedy /= n_episodes
    rewards_egreedy /= n_episodes
    plt.plot(rewards_greedy, label="greedy")
    print("Total reward of greedy strategy averaged over " + str(n_episodes) + " episodes: " + str(np.sum(rewards_greedy)))
    plt.plot(rewards_egreedy, label="e-greedy")
    print("Total reward of epsilon greedy strategy averaged over " + str(n_episodes) + " episodes: " + str(np.sum(rewards_egreedy)))
    plt.legend()
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.savefig('bandit_strategies.eps')
    plt.show()


if __name__ == "__main__":
    main()
