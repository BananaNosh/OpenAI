import gym
import four_wins_env
import tensorflow as tf
import numpy as np
env = gym.make('FourWins-v0')

spaces = env.action_space
print(spaces)
total_episodes = 1000
total_results = np.array([0, 0, 0])
for i_episode in range(total_episodes):
    _ = env.reset()
    # env.render()
    done = False
    reward = 0
    while not done:
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        # print(observation)
        # print("my rew", reward)
        # env.render()
    total_results[int(reward / 100) + 1] += 1
print(total_results / total_episodes)
