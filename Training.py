# noinspection PyPackageRequirements
import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os

from FourWins import play

def conv_net(x, target_size, activation_function=None):
    fan_in = int(x.shape[-1])
    conv_inp = tf.reshape(x, (-1, 7,6,1))
    layer = tf.layers.conv2d(conv_inp, filters=20, kernel_size=3, padding="SAME", activation=tf.nn.relu)
    print(layer)
    layer = tf.layers.conv2d(layer, filters=64, kernel_size=5, padding="SAME", activation=tf.nn.relu)
    print(layer)
    layer = tf.layers.conv2d(layer, filters=16, kernel_size=3, padding="SAME", activation=tf.nn.relu)
    print(layer)

    shapes = layer.get_shape().as_list()
    ff_input = tf.reshape(layer, (-1, shapes[1]*shapes[2]*shapes[3]))
    print(ff_input)
    dense_layer = tf.layers.dense(ff_input, 128, activation=tf.nn.relu)
    print(dense_layer)

    if activation_function == tf.nn.relu:
        var_init = tf.random_normal_initializer(stddev=2 / fan_in)
    else:
        var_init = tf.random_normal_initializer(stddev=fan_in ** (-1 / 2))
    weights = tf.get_variable("weights", [dense_layer.shape[1], target_size], tf.float32, var_init)

    var_init = tf.constant_initializer(0.0)
    biases = tf.get_variable("biases", [target_size], tf.float32, var_init)

    activation = tf.matmul(dense_layer, weights) + biases

    return activation_function(activation) if callable(activation_function) else activation, weights, biases, activation


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

    return activation_function(activation) if callable(activation_function) else activation, weights, biases, activation


def discount_rewards(rewards, _discount_factor):
    discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
    for i, reward in enumerate(reversed(rewards)):
        discounted_rewards[-(i + 1)] = discounted_rewards[-i] * _discount_factor + reward
    return discounted_rewards


class GymTraining:
    def __init__(self, env, observation_dim, hidden_layer_size, action_count,
                 observation_from_state=None, get_player_number=None):
        self.env = env
        self.input_layer_size = observation_dim
        self.hidden_layer_size = hidden_layer_size
        self.read_out_layer_size = action_count if action_count > 2 else 1
        self.action_count = action_count
        self.observation_from_state = observation_from_state if observation_from_state else lambda x: x

    def build_graph(self, decay_rate, decay_steps, initial_learning_rate, variable_scope=None):
        availability = tf.placeholder(tf.float32, [self.action_count], name="pl_availability")
        observation = tf.placeholder(tf.float32, [None, self.input_layer_size], name="pl_observation")
        global_step = tf.Variable(0, trainable=False)

        ## FOR OLD VERSION: uncomment the following and comment the three lines below
        # with tf.variable_scope("readout_layer"):
        #     readout_layer, _, _, _ = feed_forward_layer(observation, self.hidden_layer_size, tf.nn.tanh)
        #
        # with tf.variable_scope("output_layer"):
        #     action_probabilities, out_layer_weights, out_layer_biases, out_layer_activation = feed_forward_layer(readout_layer, self.read_out_layer_size, tf.nn.softmax)
        #     if self.read_out_layer_size == 1:
        #         action_probabilities = tf.concat([action_probabilities, 1 - action_probabilities], axis=1) # TODO check for 2 actions
        #     print("output:", action_probabilities)

        with tf.variable_scope("forward_pass"):
            action_probabilities, out_layer_weights, out_layer_biases, out_layer_activation = conv_net(observation, self.read_out_layer_size, tf.nn.softmax)
            print("output:", action_probabilities)

        with tf.variable_scope("action_selection"):
            gated = tf.multiply(action_probabilities, availability)
            gated = tf.divide(gated, tf.reduce_sum(gated))
            gated = tf.clip_by_value(gated, 1*10**(-30), 1)
            log_probabilities = tf.log(gated, name="log_prob")
            act_wrapped = tf.multinomial(log_probabilities, num_samples=1, name="output_action")
            act = act_wrapped[0][0]
            random_action = tf.multinomial([(availability-1)*10000], num_samples=1)[0][0] # sample some random action from all available actions (uniformly)
            decaying_prob_random = tf.train.exponential_decay(0.5, global_step, decay_steps, decay_rate) # the probability to pick the random action must be decaying
            # decaying_prob_random = tf.train.polynomial_decay(0.5, global_step, decay_steps) # can test different kinds of decay
            random_number = tf.random_uniform([1], minval=0, maxval=1)[0]
            action = tf.cond(random_number > decaying_prob_random, lambda: act, lambda: random_action) # take best action with certain probability, else take random action
            # action = act if np.random.rand()>tf.to_float(decaying_prob_random) else random_action
            log_probability = log_probabilities[:, tf.to_int32(action)]

        with tf.variable_scope("trainable_variables"):
            trainable_variables = tf.trainable_variables()
            variables_in_scope = [var for var in trainable_variables if variable_scope in var.name] \
                if variable_scope else trainable_variables

        with tf.variable_scope("gradients"):
            learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, decay_steps, decay_rate)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            gradients_and_variables = optimizer.compute_gradients(log_probability, var_list=variables_in_scope)
            gradients = [gradient * -1 for gradient, variable in gradients_and_variables]

        # Create placeholders for gradient tensors
        with tf.variable_scope("gradient_placeholder"):
            gradient_placeholders = []
            for gradient in gradients:
                gradient_placeholders.append(tf.placeholder(tf.float32, gradient.shape))

        with tf.variable_scope("training"):
            training_step = optimizer.apply_gradients(zip(gradient_placeholders, variables_in_scope),
                                                      global_step=global_step)
        return action, gradient_placeholders, gradients, observation, availability, training_step, log_probabilities, gated, action_probabilities, log_probability, out_layer_weights, out_layer_biases, out_layer_activation

    def train(self, training_episodes=40, batch_size=10, discount_factor=0.99, initial_learning_rate=0.05,
              decay_steps=100, decay_rate=0.75, render=False, save_path=None, test_play_function=None):

        tf.reset_default_graph()

        action, gradient_placeholders, gradients, observation, availability, training_step, log_probabilities, gated, action_probabilities, log_prob, out_layer_weights, out_layer_biases, out_layer_activations \
            = self.build_graph(decay_rate, decay_steps, initial_learning_rate)

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
                    print("New game started")
                    while not done:
                        # if render and ((episode % 10 == 0 and batch == 0) or episode == training_episodes - 1):
                        #     self.env.render()
                        _observation = np.reshape(self.observation_from_state(game_state), [1, -1])
                        _availability = np.array([1 if self.env.action_space.contains(i) else 0
                                                  for i in range(self.action_count)])
                        # print("availability", _availability)
                        sampled_action, sampled_gradient, _log_probabilities, _gated, _action_probabilities, _log_probability, _weights, _biases, _out_activations, _gradients = session.run(
                            [action, gradients,
                             log_probabilities, gated, action_probabilities, log_prob,
                             out_layer_weights, out_layer_biases, out_layer_activations, gradients],
                            feed_dict={
                                observation: _observation,
                                availability: _availability
                            })
                        # print("gradients", _gradients)
                        # print("act probs", _action_probabilities)
                        # print("gated", _gated)
                        # print("log probs", _log_probabilities)
                        # print("log_prob", _log_probability)
                        # print("action", sampled_action)
                        # print("weights", _weights)
                        # print("biases", _biases)
                        # print("out_activations", _out_activations)
                        sampled_gradients.append(sampled_gradient)

                        game_state, reward, done, _ = self.env.step(sampled_action)
                        # if render and ((episode % 10 == 0 and batch == 0) or episode == training_episodes - 1):
                        #     self.env.render()
                        rewards.append(reward)
                    if render:
                        self.env.render()
                    rewards_cache.append(np.sum(rewards))

                    _rewards = discount_rewards(rewards, discount_factor)
                    rewards = ((_rewards - np.mean(_rewards)) / np.std(_rewards)) \
                        if not len(np.unique(_rewards)) == 1 else np.zeros_like(_rewards)

                    # Scale gradients by normalized discounted reward
                    scaled_gradients = np.reshape(rewards, [-1, 1]) * np.array(sampled_gradients)

                    # Calculate mean gradient
                    mean_gradients.append(np.sum(scaled_gradients, axis=0))

                    mean_gradients_cache.append(mean_gradients)

                    for mean_gradient in mean_gradients:
                        feed_dict = {placeholder: mean_gradient[j]
                                     for j, placeholder in enumerate(gradient_placeholders)}
                        session.run(training_step, feed_dict=feed_dict)
                if save_path and episode % 10 == 0:
                    folder, filename = os.path.split(save_path)
                    folder = os.path.join(folder, f"e_{episode}")
                    if not os.path.exists(folder):
                        os.mkdir(folder)
                    tf.train.Saver().save(session, os.path.join(folder, filename))

            if save_path:
                saved = tf.train.Saver().save(session, save_path)
                print("Model saved in path: %s" % saved)
            if test_play_function is not None:
                ai_plays = lambda _observation, _availability: session.run(
                    [action],
                    feed_dict={
                        observation: np.reshape(_observation, [1, -1]),
                        availability: _availability
                    })[0]
                test_play_function(self.env, ai_plays)

        self.env.close()

        print(rewards_cache)
        fig, axes = plt.subplots(2, 2)

        axes[0, 0].set_title("Rewards")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Reward")

        colors = ["b", "r", "g", "y"]
        color = colors[0 % len(colors)]
        axes[0, 0].plot(np.mean(np.array(rewards_cache).reshape(-1, batch_size), axis=1), color=color)
        axes[0, 1].plot(rewards_cache, color=color)
        axes[1, 0].plot(np.sum(np.array(rewards_cache).reshape(-1, batch_size) > 0, axis=1), color=color)
        plt.show()

        # checkpoints = [os.path.join("saved_models", f"e_{e}") for e in range(0, training_episodes, 10)]
        # for c1, c2 in zip(checkpoints[1:], checkpoints[:-1]):
        #     win_ratios.append(evaluate_checkpoints(c1, c2))
        # print(win_ratios)
        # plt.plot(win_ratios)


if __name__ == '__main__':
    # training = GymTraining(gym.make("CartPole-v0"), 4, 8, 2)
    # training.train(save_path="./saved_models/cart_pole_model.ckpt", training_episodes=40)
    env = gym.make("FourWins-v0")
    env.play_both = True
    training = GymTraining(env, 42, 84, 7, observation_from_state=lambda x: x[0])
    training.train(training_episodes=200, batch_size=10, discount_factor=0.99, initial_learning_rate=0.005,
                   save_path="./saved_models/four_wins_model_one_player.ckpt", render=True, test_play_function=play)
