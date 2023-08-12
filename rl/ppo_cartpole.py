import gymnasium as gym
import numpy as np
import scipy as sp
import tensorflow as tf
import time


# Hyperparameters

epochs = 30
steps_for_epoch = 4000
clip_ratio = 0.2
gamma = 0.99
policy_learning_rate = 3e-4
value_function_learning_rate = 1e-3
train_policy_iterations = 80
train_value_iterations = 80
lam = 0.97
target_kl = 0.01
hidden_sizes = (64, 64)

render = True


# Functions and class

def discounted_cumulative_sums(x, discount):
    """
    Discounted cumulative sums of vectors for computing rewards-to-go
    and advantage estimates
    """
    return sp.signal.lfilter(
        b=[1], a=[1, float(-discount)], x=x[::-1], axis=0
    )[::-1]


class Buffer:
    """
    Buffer for storing trajectories
    """

    def __init__(
        self,
        arg_observation_dimensions,
        arg_size,
        arg_gamma=0.99,
        arg_lam=0.95
    ):
        """
        Buffer initialization
        """
        self.observation_buffer = np.zeros(
            shape=(arg_size, arg_observation_dimensions), dtype=np.float32
        )
        self.action_buffer = np.zeros(shape=arg_size, dtype=np.int32)
        self.advantage_buffer = np.zeros(shape=arg_size, dtype=np.float32)
        self.reward_buffer = np.zeros(shape=arg_size, dtype=np.float32)
        self.return_buffer = np.zeros(shape=arg_size, dtype=np.float32)
        self.value_buffer = np.zeros(shape=arg_size, dtype=np.float32)
        self.logprobability_buffer = np.zeros(shape=arg_size, dtype=np.float32)
        self.gamma, self.lam = arg_gamma, arg_lam
        self.pointer, self.trajectory_start_index = 0, 0

    def store(
        self,
        arg_observation,
        arg_action,
        arg_reward,
        arg_value,
        arg_logprobability
    ):
        """
        Append one step of agent-environment interaction
        """
        self.observation_buffer[self.pointer] = arg_observation
        self.action_buffer[self.pointer] = arg_action
        self.reward_buffer[self.pointer] = arg_reward
        self.value_buffer[self.pointer] = arg_value
        self.logprobability_buffer[self.pointer] = arg_logprobability
        self.pointer += 1

    def finish_trajectory(self, arg_last_value=0):
        """
        Finish the trajectory by computing advantage estimates and rewards-to-go
        """
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], arg_last_value)
        values = np.append(self.value_buffer[path_slice], arg_last_value)
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.advantage_buffer[path_slice] = discounted_cumulative_sums(
            deltas, self.gamma * self.lam
        )
        self.return_buffer[path_slice] = discounted_cumulative_sums(
            rewards, self.gamma
        )[:-1]
        self.trajectory_start_index = self.pointer

    def get(self):
        """
        Get all data of the buffer and normalize the advantages
        """
        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean, advantage_std = (
            np.mean(self.advantage_buffer), np.std(self.advantage_buffer)
        )
        self.advantage_buffer = (
            (self.advantage_buffer - advantage_mean) / advantage_std
        )
        return (
            self.observation_buffer,
            self.action_buffer,
            self.advantage_buffer,
            self.return_buffer,
            self.logprobability_buffer
        )


def mlp(x, sizes, activation=tf.tanh, output_activation=None):
    """
    Build a feedforward neural network
    """
    for size in sizes[:-1]:
        x = tf.keras.layers.Dense(units=size, activation=activation)(x)
    return tf.keras.layers.Dense(units=sizes[-1], activation=output_activation)(x)


def logprobabilities(arg_logits, arg_a):
    """
    Compute the log-probabilities of taking actions a by using the logits
    (i.e. the output of the actor)
    """
    logprobabilities_all = tf.nn.log_softmax(arg_logits)
    logprobability = tf.reduce_sum(
        tf.one_hot(arg_a, num_actions) * logprobabilities_all, axis=1
    )
    return logprobability


# Initializations

env = gym.make(id="CartPole-v1", render_mode="human")
observation_dimensions = env.observation_space.shape[0]
num_actions = env.action_space.n

buffer = Buffer(observation_dimensions, steps_for_epoch)

observation_input = tf.keras.Input(
    shape=(observation_dimensions,), dtype=tf.float32
)
logits = mlp(observation_input, list(hidden_sizes) + [num_actions])
actor = tf.keras.Model(inputs=observation_input, outputs=logits)
value = tf.squeeze(
    input=mlp(observation_input, list(hidden_sizes) + [1]), axis=1
)
critic = tf.keras.Model(inputs=observation_input, outputs=value)

policy_optimizer = tf.optimizers.Adam(learning_rate=policy_learning_rate)
value_optimizer = tf.optimizers.Adam(learning_rate=value_function_learning_rate)

observation, episode_return, episode_length = env.reset()[0], 0, 0


# Train functions

@tf.function
def sample_action(arg_observation):
    """
    Sample action from actor
    """
    _logits = actor(arg_observation)
    _action = tf.squeeze(tf.random.categorical(_logits, num_samples=1), axis=1)
    return _logits, _action


@tf.function
def train_policy(
    arg_observation_buffer,
    arg_action_buffer,
    arg_logprobability_buffer,
    arg_advantage_buffer
):
    """
    Train the policy by maximizing the PPO-Clip objective
    """
    with tf.GradientTape() as tape:
        ratio = tf.exp(
            logprobabilities(actor(arg_observation_buffer), arg_action_buffer)
            - arg_logprobability_buffer
        )
        min_advantage = tf.where(
            arg_advantage_buffer > 0,
            (1 + clip_ratio) * arg_advantage_buffer,
            (1 - clip_ratio) * arg_advantage_buffer,
        )
        policy_loss = -tf.reduce_mean(
            tf.minimum(ratio * arg_advantage_buffer, min_advantage)
        )
    policy_grads = tape.gradient(policy_loss, actor.trainable_variables)
    policy_optimizer.apply_gradients(zip(policy_grads, actor.trainable_variables))
    _kl = tf.reduce_mean(
        arg_logprobability_buffer
        - logprobabilities(actor(arg_observation_buffer), arg_action_buffer)
    )
    _kl = tf.reduce_sum(_kl)
    return _kl


@tf.function
def train_value_function(arg_observation_buffer, arg_return_buffer):
    """
    Train the value function by regression on mean-squared error
    """
    with tf.GradientTape() as tape:
        value_loss = tf.reduce_mean(
            (arg_return_buffer - critic(arg_observation_buffer)) ** 2
        )
    value_grads = tape.gradient(value_loss, critic.trainable_variables)
    value_optimizer.apply_gradients(zip(value_grads, critic.trainable_variables))


# Train

for epoch in range(epochs):
    t0 = time.time()
    sum_return = 0
    sum_length = 0
    num_episodes = 0

    # Iterate over the steps of each epoch
    for t in range(steps_for_epoch):
        if render:
            env.render()

        observation = observation.reshape(1, -1)
        logits, action = sample_action(observation)
        observation_new, reward, terminated, _, _ = env.step(action[0].numpy())
        episode_return += reward
        episode_length += 1

        value_t = critic(observation)
        logprobability_t = logprobabilities(logits, action)

        buffer.store(observation, action, reward, value_t, logprobability_t)

        observation = observation_new

        terminal = terminated
        if terminal or (t == steps_for_epoch - 1):
            last_value = 0 if terminated else critic(observation.reshape(1, -1))
            buffer.finish_trajectory(last_value)
            sum_return += episode_return
            sum_length += episode_length
            num_episodes += 1
            observation, episode_return, episode_length = env.reset()[0], 0, 0

    # Get values from the buffer
    (
        observation_buffer,
        action_buffer,
        advantage_buffer,
        return_buffer,
        logprobability_buffer
    ) = buffer.get()

    # Update the policy and implement early stopping using KL divergence
    for _ in range(train_policy_iterations):
        kl = train_policy(
            arg_observation_buffer=observation_buffer,
            arg_action_buffer=action_buffer,
            arg_logprobability_buffer=logprobability_buffer,
            arg_advantage_buffer=advantage_buffer
        )
        if kl > 1.5 * target_kl:
            break

    # Update the value function
    for _ in range(train_value_iterations):
        train_value_function(
            arg_observation_buffer=observation_buffer,
            arg_return_buffer=return_buffer
        )

    # Print mean return and length for each epoch
    print(
        f" Epoch: {epoch + 1}."
        f" Mean return: {sum_return / num_episodes:.2f}."
        f" Mean length: {sum_length / num_episodes:.2f}."
        f" Time cost: {time.time() - t0:.0f} seconds."
    )

#  Epoch: 1. Mean return: 16.13. Mean length: 16.13. Time cost: 93 seconds.
#  Epoch: 2. Mean return: 19.80. Mean length: 19.80. Time cost: 85 seconds.
#  Epoch: 3. Mean return: 26.49. Mean length: 26.49. Time cost: 84 seconds.
#  Epoch: 4. Mean return: 32.26. Mean length: 32.26. Time cost: 86 seconds.
#  Epoch: 5. Mean return: 41.24. Mean length: 41.24. Time cost: 84 seconds.
#  Epoch: 6. Mean return: 67.80. Mean length: 67.80. Time cost: 82 seconds.
#  Epoch: 7. Mean return: 88.89. Mean length: 88.89. Time cost: 82 seconds.
#  Epoch: 8. Mean return: 137.93. Mean length: 137.93. Time cost: 82 seconds.
#  Epoch: 9. Mean return: 166.67. Mean length: 166.67. Time cost: 82 seconds.
#  Epoch: 10. Mean return: 235.29. Mean length: 235.29. Time cost: 82 seconds.
#  Epoch: 11. Mean return: 363.64. Mean length: 363.64. Time cost: 83 seconds.
#  Epoch: 12. Mean return: 363.64. Mean length: 363.64. Time cost: 82 seconds.
#  Epoch: 13. Mean return: 444.44. Mean length: 444.44. Time cost: 82 seconds.
#  Epoch: 14. Mean return: 500.00. Mean length: 500.00. Time cost: 82 seconds.
#  Epoch: 15. Mean return: 1000.00. Mean length: 1000.00. Time cost: 82 seconds.
#  Epoch: 16. Mean return: 500.00. Mean length: 500.00. Time cost: 82 seconds.
#  Epoch: 17. Mean return: 2000.00. Mean length: 2000.00. Time cost: 82 seconds.
#  Epoch: 18. Mean return: 4000.00. Mean length: 4000.00. Time cost: 82 seconds.
#  Epoch: 19. Mean return: 4000.00. Mean length: 4000.00. Time cost: 81 seconds.
#  Epoch: 20. Mean return: 4000.00. Mean length: 4000.00. Time cost: 81 seconds.
#  Epoch: 21. Mean return: 4000.00. Mean length: 4000.00. Time cost: 82 seconds.
#  Epoch: 22. Mean return: 4000.00. Mean length: 4000.00. Time cost: 82 seconds.
#  Epoch: 23. Mean return: 4000.00. Mean length: 4000.00. Time cost: 83 seconds.
#  Epoch: 24. Mean return: 4000.00. Mean length: 4000.00. Time cost: 82 seconds.
#  Epoch: 25. Mean return: 4000.00. Mean length: 4000.00. Time cost: 82 seconds.
#  Epoch: 26. Mean return: 4000.00. Mean length: 4000.00. Time cost: 82 seconds.
#  Epoch: 27. Mean return: 4000.00. Mean length: 4000.00. Time cost: 82 seconds.
#  Epoch: 28. Mean return: 4000.00. Mean length: 4000.00. Time cost: 81 seconds.
#  Epoch: 29. Mean return: 4000.00. Mean length: 4000.00. Time cost: 82 seconds.
#  Epoch: 30. Mean return: 4000.00. Mean length: 4000.00. Time cost: 82 seconds.
