import tensorflow as tf
import gym

# print("\n".join(str(gym.envs.registry.all()).split(",")))
env = gym.make("Taxi-v2")
# observation = env.reset()
spaces = env.action_space
# print(observation)
print(spaces)
for i_episode in range(10):
    observation = env.reset()
    for t in range(1000):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if t % 100 == 0:
            print(observation)
            print(reward)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
        if t == 99:
            print("Did not finish")