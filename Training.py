import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
import four_wins_env


def feed_forward_layer(x, target_size, activation_function=None):
    print("Forward-Layer:" + str(x.shape))

    fan_in = int(x.shape[-1])

    if activation_function == tf.nn.relu:
        var_init = tf.random_normal_initializer(stddev=2 / fan_in)
    else:
        var_init = tf.random_normal_initializer(stddev=fan_in ** (-1 / 2))
    weights = tf.get_variable("weights", [x.shape[1], target_size], tf.float32, var_init)

    var_init = tf.constant_initializer(0.0)
    biases = tf.get_variable("biases", [target_size], tf.float32, var_init)

    activation = tf.matmul(x, weights) + biases

    return activation_function(activation) if callable(activation_function) else activation


def discount_rewards(rewards, _discount_factor):
    discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
    for i, reward in enumerate(reversed(rewards)):
        discounted_rewards[-(i + 1)] = discounted_rewards[-i] * _discount_factor + reward
    return discounted_rewards


class GymTraining:
    def __init__(self, env, observation_dim, hidden_layer_size, action_count, filter_observation=None):
        self.env = env
        self.input_layer_size = observation_dim
        self.hidden_layer_size = hidden_layer_size
        self.read_out_layer_size = action_count if action_count > 2 else 1
        self.filter_observation = filter_observation if filter_observation else lambda x: x

    def build_graph(self, decay_rate, decay_steps, initial_learning_rate):
        tf.reset_default_graph()

        observation = tf.placeholder(tf.float32, [None, self.input_layer_size])
        global_step = tf.Variable(0, trainable=False)

        with tf.variable_scope("readout_layer"):
            readout_layer = feed_forward_layer(observation, self.hidden_layer_size, tf.nn.tanh)

        with tf.variable_scope("output_layer"):
            action_probabilities = feed_forward_layer(readout_layer, self.read_out_layer_size, tf.nn.sigmoid)
            if self.read_out_layer_size == 1:
                action_probabilities = tf.concat([action_probabilities, 1 - action_probabilities], axis=1)
            print(action_probabilities)

        with tf.variable_scope("action_selection"):
            log_probabilities = tf.log(action_probabilities)
            action = tf.multinomial(log_probabilities, num_samples=1)[0][0]
            log_probability = log_probabilities[:, tf.to_int32(action)]

        with tf.variable_scope("gradients"):
            learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, decay_steps, decay_rate)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            gradients_and_variables = optimizer.compute_gradients(log_probability)
            gradients = [gradient_and_variable[0] * -1 for gradient_and_variable in gradients_and_variables]

        # Create placeholders for gradient tensors
        with tf.variable_scope("gradient_placeholder"):
            gradient_placeholders = []
            for gradient in gradients:
                gradient_placeholders.append(tf.placeholder(tf.float32, gradient.shape))

        with tf.variable_scope("training"):
            trainable_variables = tf.trainable_variables()
            training_step = optimizer.apply_gradients(zip(gradient_placeholders, trainable_variables),
                                                      global_step=global_step)
        return action, gradient_placeholders, gradients, observation, training_step

    def train(self, training_episodes=40, batch_size=10, discount_factor=0.99, initial_learning_rate=0.05,
              decay_steps=100, decay_rate=0.75, render=False, save_path=None):

        action, gradient_placeholders, gradients, observation, training_step = self.build_graph(decay_rate, decay_steps,
                                                                                                initial_learning_rate)
        rewards_cache = []
        mean_gradients_cache = []

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())

            for episode in range(training_episodes):
                print("episode:", episode)
                mean_gradients = []

                for batch in range(batch_size):
                    done = False
                    rewards = []
                    sampled_gradients = []

                    game_state = self.env.reset()

                    while not done:
                        # if render and ((episode % 10 == 0 and batch == 0) or episode == training_episodes - 1):
                        #     self.env.render()

                        _observation = np.reshape(self.filter_observation(game_state), [1, -1])
                        sampled_action, sampled_gradient = session.run(
                            [action, gradients],
                            feed_dict={
                                observation: _observation
                            })

                        sampled_gradients.append(sampled_gradient)

                        game_state, reward, done, _ = self.env.step(sampled_action)
                        # if render and ((episode % 10 == 0 and batch == 0) or episode == training_episodes - 1):
                        #     self.env.render()
                        rewards.append(reward)
                    self.env.render()
                    rewards_cache.append(np.sum(rewards))

                    rewards = discount_rewards(rewards, discount_factor)
                    rewards = ((rewards - np.mean(rewards)) / np.std(rewards)) if not len(np.unique(rewards)) == 1 \
                        else np.zeros_like(rewards)

                    # Scale gradients by normalized discounted reward
                    scaled_gradients = np.reshape(rewards, [-1, 1]) * np.array(sampled_gradients)

                    # Calculate mean gradient
                    mean_gradients.append(np.sum(scaled_gradients, axis=0))

                    mean_gradients_cache.append(mean_gradients)

                for mean_gradient in mean_gradients:
                    feed_dict = {placeholder: mean_gradient[i] for i, placeholder in enumerate(gradient_placeholders)}
                    session.run(training_step, feed_dict=feed_dict)
            if save_path:
                saved = tf.train.Saver().save(session, save_path)
                print("Model saved in path: %s" % saved)
        self.env.close()

        print(rewards_cache)
        fig, axes = plt.subplots(2, 2)

        axes[0, 0].set_title("Rewards")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Reward")

        axes[0, 0].plot(np.mean(np.array(rewards_cache).reshape(-1, batch_size), axis=1))
        axes[0, 1].plot(rewards_cache)
        axes[1, 0].plot(np.sum(np.array(rewards_cache).reshape(-1, batch_size) > 0, axis=1))
        plt.show()


if __name__ == '__main__':
    # training = GymTraining(gym.make("CartPole-v0"), 4, 8, 2)
    # training.train(save_path="./saved_models/cart_pole_model.ckpt")
    env = gym.make("FourWins-v0")
    # env.play_both = True
    training = GymTraining(env, 42, 84, 7, lambda x: x[0])
    training.train(training_episodes=200, batch_size=10, discount_factor=1,
                   save_path="./saved_models/four_wins_model.ckpt", render=True)
