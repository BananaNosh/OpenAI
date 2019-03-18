import gym
import numpy as np
from four_wins_env.four_wins import COLUMN_COUNT


def play(env, ai_plays=None):
    total_episodes = 1000
    total_results = np.array([0, 0, 0])
    for i_episode in range(total_episodes):
        observation = env.reset()
        env.render()
        done = False
        while not done:
            if observation[1] == 0:
                if ai_plays is not None:
                    availability = np.array([1 if env.action_space.contains(i) else 0 for i in range(COLUMN_COUNT)])
                    action = ai_plays(observation[0], availability)
                else:
                    action = env.action_space.sample()
            else:
                print("human's turn:")
                while True:
                    try:
                        action = int(input("human's turn - Which column? "))
                        action -= 1
                        if 0 <= action < COLUMN_COUNT:
                            break
                        else:
                            print("Please enter an integer between 1 and 7")
                    except ValueError:
                        print("Please enter an integer between 1 and 7")
            observation, reward, done, info = env.step(action)
            # print(observation)
            env.render()
            if done:
                print(f"You {'draw' if reward == 0 else 'won' if observation[1] == 1 else 'lost'}!")
                stop = input("Continue? Empty for stop:")
                if len(stop) == 0:
                    return
        # total_results[int(reward / 100) + 1] += 1
    # print(total_results / total_episodes)


if __name__ == '__main__':
    env = gym.make('FourWins-v0')
    env.play_both = True
    play(env)
