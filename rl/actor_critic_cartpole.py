import gymnasium as gym
import numpy as np
import tensorflow as tf


# Setup

seed = 42
gamma = 0.99  # Discount factor for past rewards
max_steps_per_episode = 10000
env = gym.make(id="CartPole-v1", render_mode="human")  # Create the environment
env.reset(seed=seed)
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0


# Implement Actor Critic network

num_inputs = 4
num_actions = 2
num_hidden = 128

inputs = tf.keras.Input(shape=(num_inputs,))
common = tf.keras.layers.Dense(num_hidden, activation="relu")(inputs)
action = tf.keras.layers.Dense(num_actions, activation="softmax")(common)
critic = tf.keras.layers.Dense(1)(common)

model = tf.keras.Model(inputs=inputs, outputs=[action, critic])


# Train

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
huber_loss = tf.keras.losses.Huber()
action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0

while True:
    state = env.reset()[0]
    episode_reward = 0
    with tf.GradientTape() as tape:
        for timestep in range(1, max_steps_per_episode):
            env.render()

            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, axis=0)

            # Predict action probabilities and estimated future rewards
            # from environment state
            action_probs, critic_value = model(state)
            critic_value_history.append(critic_value[0, 0])

            # Sample action from action probability distribution
            action = np.random.choice(num_actions, p=np.squeeze(action_probs))
            action_probs_history.append(tf.math.log(action_probs[0, action]))

            # Apply the sampled action in our environment
            state, reward, terminated, _, _ = env.step(action=action)
            rewards_history.append(reward)
            episode_reward += reward

            if terminated:
                break

        # Update running reward to check condition for solving
        running_reward = 0.05 * episode_reward + 0.95 * running_reward

        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are the labels for our critic
        rewards = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = float(r) + gamma * discounted_sum
            rewards.insert(0, discounted_sum)

        # Normalize
        rewards = np.array(rewards)
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + eps)
        rewards = rewards.tolist()

        # Calculating loss values to update our network
        history = zip(action_probs_history, critic_value_history, rewards)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log
            # probability of `log_prob` and ended up recieving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff = ret - value
            actor_losses.append(-log_prob * diff)

            # The critic must be updated so that it predicts a better estimate of
            # the future rewards.
            critic_losses.append(
                huber_loss(tf.expand_dims(value, axis=0), tf.expand_dims(ret, axis=0))
            )

        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Clear the loss and reward history
        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()

    # Log details
    episode_count += 1
    if episode_count % 10 == 0:
        print(f"running reward: {running_reward:.2f} at episode {episode_count}")

    if running_reward > 195:  # Condition to consider the task solved
        print(f"Solved at episode {episode_count}!")
        break

# running reward: 14.77 at episode 10
# running reward: 27.91 at episode 20
# running reward: 45.89 at episode 30
# running reward: 55.67 at episode 40
# running reward: 71.75 at episode 50
# running reward: 87.22 at episode 60
# running reward: 100.68 at episode 70
# Solved at episode 78!
