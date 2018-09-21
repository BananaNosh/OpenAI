import gym
import four_wins_env
import tensorflow as tf
import numpy as np


env = gym.make('FourWins-v0')


def play(both=False, human=False):
    env.play_both = both
    total_episodes = 1000
    total_results = np.array([0, 0, 0])
    for i_episode in range(total_episodes):
        observation = env.reset()
        env.render()
        done = False
        reward = 0
        while not done:
            if observation[1] == 0:
                action = env.action_space.sample()
            elif human:
                print("human's turn:")
                while True:
                    try:
                        action = int(input("human's turn - Which column? "))
                        action -= 1
                        if 0 <= action <= 6:
                            break
                        else:
                            print("Please enter an integer between 1 and 7")
                    except ValueError:
                        print("Please enter an integer between 1 and 7")
            else:
                raise AssertionError("Not yet implemented")
            observation, reward, done, info = env.step(action)
            # print(observation)
            env.render()
            print("rew", reward)
            if human and done:
                print(f"You {'draw' if reward == 0 else 'won' if observation[1] == 1 else 'lost'}!")
                stop = input("Continue? Empty for stop:")
                if len(stop) == 0:
                    return
        # total_results[int(reward / 100) + 1] += 1
    # print(total_results / total_episodes)


if __name__ == '__main__':
    play(True, True)
