import gym
import os
import tensorflow as tf
import numpy as np
from four_wins_env.four_wins import COLUMN_COUNT


def evaluate_checkpoints(c1, c2, filename="four_wins_model_one_player.ckpt"):
    env = gym.make("FourWins-v0")
    env.play_both = True
    c_meta = os.path.join(c1, f"{filename}.meta")
    # tf.reset_default_graph()
    saver = tf.train.import_meta_graph(c_meta)
    sessions = []
    for i, c in enumerate([c1, c2]):
        sessions.append(tf.Session())
        saver.restore(sessions[i], os.path.join(c, filename))

    graph = tf.get_default_graph()

    in_availability = graph.get_tensor_by_name("pl_availability:0")
    in_observation = graph.get_tensor_by_name("pl_observation:0")
    output_action = graph.get_tensor_by_name("action_selection/output_action/Multinomial:0")[0][0]

    wins = [0, 0]
    game_count = 1000
    for i in range(game_count):
        player = np.random.choice(2)
        game_state = env.reset()
        while True:
            _observation = np.reshape(game_state[0], [1, -1])
            _availability = np.array([1 if env.action_space.contains(i) else 0
                                      for i in range(COLUMN_COUNT)])
            # output_action = graph.get_collection("output_action:0")
            sampled_action = sessions[player].run(output_action, feed_dict={in_observation: _observation, in_availability: _availability})
            game_state, reward, done, _ = env.step(sampled_action)
            if done:
                print(i+1, "done", wins)
                if reward > 0:
                    wins[player] += 1
                break
            player = (player + 1) % 2
    return wins[0]/wins[1] if wins[1] > 0 else game_count


if __name__ == '__main__':
    saved_models_path = "saved_models"
    print(evaluate_checkpoints(os.path.join(saved_models_path, "e_20"), os.path.join(saved_models_path, "e_0")))