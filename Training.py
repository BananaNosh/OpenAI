# noinspection PyPackageRequirements
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

    return activation_function(activation) if callable(activation_function) else activation, weights, biases, activation


def discount_rewards(rewards, _discount_factor):
    discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
    for i, reward in enumerate(reversed(rewards)):
        discounted_rewards[-(i + 1)] = discounted_rewards[-i] * _discount_factor + reward
    return discounted_rewards


class GymTraining:
    def __init__(self, env, observation_dim, hidden_layer_size, action_count, player_count=1,
                 observation_from_state=None, get_player_number=None):
        self.env = env
        self.input_layer_size = observation_dim
        self.hidden_layer_size = hidden_layer_size
        self.read_out_layer_size = action_count if action_count > 2 else 1
        self.action_count = action_count
        self.player_count = player_count
        self.observation_from_state = observation_from_state if observation_from_state else lambda x: x
        self.get_player_number = get_player_number if get_player_number else lambda x: 0

    def build_graph(self, decay_rate, decay_steps, initial_learning_rate, variable_scope=None):
        availability = tf.placeholder(tf.float32, [self.action_count])
        observation = tf.placeholder(tf.float32, [None, self.input_layer_size])
        global_step = tf.Variable(0, trainable=False)

        with tf.variable_scope("readout_layer"):
            readout_layer, _, _, _ = feed_forward_layer(observation, self.hidden_layer_size, tf.nn.tanh)

        with tf.variable_scope("output_layer"):
            action_probabilities, out_layer_weights, out_layer_biases, out_layer_activation = feed_forward_layer(readout_layer, self.read_out_layer_size, tf.nn.softmax)
            if self.read_out_layer_size == 1:
                action_probabilities = tf.concat([action_probabilities, 1 - action_probabilities], axis=1) # TODO check for 2 actions
            print("output:", action_probabilities)

        with tf.variable_scope("action_selection"):
            gated = tf.multiply(action_probabilities, availability)
            gated = tf.divide(gated, tf.reduce_sum(gated))
            maximum = tf.reduce_min(tf.boolean_mask(gated, tf.greater(gated, 0)), axis=0)
            gated = tf.clip_by_value(gated, 1*10**(-30), 1)
            log_probabilities = tf.log(gated)
            # log_availability = tf.log(tf.clip_by_value(availability+maximum/1000000, 0, 1))
            # gated_probs = tf.divide(gated, tf.reduce_sum(gated))
            act = tf.multinomial(log_probabilities, num_samples=1)[0][0]
            random_action = tf.multinomial([(availability-1)*10000], num_samples=1)[0][0] # sample some random action from all available actions (uniformly)
            decaying_prob_random = tf.train.exponential_decay(0.5, global_step, decay_steps, decay_rate) # the probability to pick the random action must be decaying
            # decaying_prob_random = tf.train.polynomial_decay(0.5, global_step, decay_steps) # can test different kinds of decay
            random_number = tf.random_uniform([1], minval=0, maxval=1)[0]
            action = tf.cond(random_number > decaying_prob_random, lambda: act, lambda: random_action) # take best action with certain probability, else take random action
            # action = act if np.random.rand()>tf.to_float(decaying_prob_random) else random_action
            log_probability = log_probabilities[:, tf.to_int32(action)]
            # gated_probability = gated_probs[:, tf.to_int32(action)]

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
              decay_steps=100, decay_rate=0.75, render=False, save_path=None):

        tf.reset_default_graph()

        # build a network for every player
        action = []
        gradient_placeholders = []
        gradients = []
        observation = []
        availabilities = []
        training_step = []
        log_probs = []
        gated = []
        action_probs = []
        log_probability = []
        out_layer_weights = []
        out_layer_biases = []
        out_layer_activations = []
        for i in range(self.player_count):
            scope = f"player_{i}"
            with tf.variable_scope(scope):
                _action, _gradient_placeholders, _gradients, _observation, _availability, _training_step, _log_probabilities, _gated, _action_probabilities, _log_prob, _out_layer_weights, _out_layer_biases, _out_layer_activations \
                    = self.build_graph(decay_rate, decay_steps, initial_learning_rate, variable_scope=scope)
                action.append(_action)
                gradient_placeholders.append(_gradient_placeholders)
                gradients.append(_gradients)
                observation.append(_observation)
                availabilities.append(_availability)
                training_step.append(_training_step)
                log_probs.append(_log_probabilities)
                gated.append(_gated)
                action_probs.append(_action_probabilities)
                log_probability.append(_log_prob)
                out_layer_weights.append(_out_layer_weights)
                out_layer_biases.append(_out_layer_biases)
                out_layer_activations.append(_out_layer_activations)

        rewards_cache = [[] for _ in range(self.player_count)]
        mean_gradients_cache = [[] for _ in range(self.player_count)]

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())

            for episode in range(training_episodes):
                print("episode:", episode)
                mean_gradients = [[] for _ in range(self.player_count)]

                for batch in range(batch_size):
                    done = False
                    rewards = [[] for _ in range(self.player_count)]
                    sampled_gradients = [[] for _ in range(self.player_count)]

                    game_state = self.env.reset()
                    print("New game started")
                    while not done:
                        # if render and ((episode % 10 == 0 and batch == 0) or episode == training_episodes - 1):
                        #     self.env.render()

                        _observation = np.reshape(self.observation_from_state(game_state), [1, -1])
                        _availability = np.array([1 if self.env.action_space.contains(i) else 0
                                                  for i in range(self.action_count)])
                        print("availability", _availability)
                        current_player = self.get_player_number(game_state)
                        sampled_action, sampled_gradient, _log_probabilities, _gated, _action_probabilities, _log_probability, _weights, _biases, _out_activations, _gradients = session.run(
                            [action[current_player], gradients[current_player],
                             log_probs[current_player], gated[current_player], action_probs[current_player], log_probability[current_player],
                             out_layer_weights[current_player], out_layer_biases[current_player], out_layer_activations[current_player], gradients[current_player]],
                            feed_dict={
                                observation[current_player]: _observation,
                                availabilities[current_player]: _availability
                            })
                        # print("gradients", _gradients)
                        print("act probs", _action_probabilities)
                        print("gated", _gated)
                        print("log probs", _log_probabilities)
                        print("log_prob", _log_probability)
                        print("action", sampled_action)
                        # print("weights", _weights)
                        # print("biases", _biases)
                        # print("out_activations", _out_activations)
                        sampled_gradients[current_player].append(sampled_gradient)

                        game_state, reward, done, _ = self.env.step(sampled_action)
                        # if render and ((episode % 10 == 0 and batch == 0) or episode == training_episodes - 1):
                        #     self.env.render()
                        rewards[current_player].append(reward)
                        if done:
                            for i in range(self.player_count):
                                if i != current_player:
                                    rewards[i][-1] -= reward
                    if render:
                        self.env.render()
                    for i in range(self.player_count):
                        rewards_cache[i].append(np.sum(rewards[i]))

                        _rewards = discount_rewards(rewards[i], discount_factor)
                        rewards[i] = ((_rewards - np.mean(_rewards)) / np.std(_rewards)) \
                            if not len(np.unique(_rewards)) == 1 else np.zeros_like(_rewards)

                        # Scale gradients by normalized discounted reward
                        scaled_gradients = np.reshape(rewards[i], [-1, 1]) * np.array(sampled_gradients[i])

                        # Calculate mean gradient
                        mean_gradients[i].append(np.sum(scaled_gradients, axis=0))

                        mean_gradients_cache[i].append(mean_gradients[i])

                        for mean_gradient in mean_gradients[i]:
                            feed_dict = {placeholder: mean_gradient[j]
                                         for j, placeholder in enumerate(gradient_placeholders[i])}
                            session.run(training_step[i], feed_dict=feed_dict)
            if save_path:
                saved = tf.train.Saver().save(session, save_path)
                print("Model saved in path: %s" % saved)
        self.env.close()

        print(rewards_cache)
        fig, axes = plt.subplots(2, 2)

        axes[0, 0].set_title("Rewards")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Reward")

        colors = ["b", "r", "g", "y"]
        for i in range(self.player_count):
            color = colors[i % len(colors)]
            axes[0, 0].plot(np.mean(np.array(rewards_cache[i]).reshape(-1, batch_size), axis=1), color=color)
            axes[0, 1].plot(rewards_cache[i], color=color)
            axes[1, 0].plot(np.sum(np.array(rewards_cache[i]).reshape(-1, batch_size) > 0, axis=1), color=color)
        plt.show()


if __name__ == '__main__':
    # training = GymTraining(gym.make("CartPole-v0"), 4, 8, 2)
    # training.train(save_path="./saved_models/cart_pole_model.ckpt", training_episodes=40)
    env = gym.make("FourWins-v0")
    env.play_both = True
    training = GymTraining(env, 42, 84, 7, player_count=1, observation_from_state=lambda x: x[0])
    training.train(training_episodes=20, batch_size=10, discount_factor=0.99, initial_learning_rate=0.005,
                   save_path="./saved_models/four_wins_model_one_player.ckpt", render=True)
    # number = 1
    # while True:
    #     loga = np.log(number)
    #     print(number, loga)
    #     number /= 10
    #     if number == 0:
    #         break
