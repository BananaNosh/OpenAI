import gym
import four_wins_env
env = gym.make('FourWins-v0')

spaces = env.action_space
# print(observation)
print(spaces)
for i_episode in range(1):
    observation = env.reset()
    for t in range(42):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        # print(observation)
        print(reward)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break