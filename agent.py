import tensorflow as tf
import keras
import numpy as np
from actor_critic import ActorCritic
from env import GcsimEnv
from tqdm import tqdm
from custom_plot import plot_time_series

# TODO:
# could not execute skill; action not ready
# 

class Agent:
    def __init__(self, env: GcsimEnv, gamma=0.9, alpha=7e-4, entropy_coeff = 0.01):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.entropy_coeff = entropy_coeff
        
        self.state_size = self.env.get_state_size()
        self.n_actions = self.env.get_n_actions()
        self.actor_critic = ActorCritic(self.state_size, self.n_actions)
        # self.actor_critic.compile(optimizer=keras.optimizers.Adam(learning_rate=alpha))

        self.cd_knowledge = np.array([1.5], dtype=np.float32)
        self.cd_knowledge = tf.convert_to_tensor([self.cd_knowledge], dtype=tf.float32)
        self.optimizer_shared = keras.optimizers.Adam(learning_rate=7e-4)
        self.optimizer_actor  = keras.optimizers.Adam(learning_rate=7e-4)
        self.optimizer_critic = keras.optimizers.Adam(learning_rate=7e-4)

    def predict(self, state):
        action, _, _ = self._predict(state)
        return action

    def _predict(self, state: np.ndarray):    
        state = tf.convert_to_tensor(state.reshape((1, -1)), dtype=tf.int32)
        prob_distribution, state_value = self.actor_critic((state, self.cd_knowledge))

        sampled_action_index = np.random.choice(self.n_actions, p=prob_distribution.numpy()[0])
        action = sampled_action_index + 1 # used in env.step()

        action_prob = prob_distribution[0, sampled_action_index]
        state_value = state_value[0, 0]

        return action, action_prob, state_value, prob_distribution

    def learn(self, n_episodes=1000):
        reward_end_of_episode = []
        for episode in range(1, n_episodes+1):
            print("episode", episode)
            
            state = self.env.reset()
            done = False

            while not done:
                with tf.GradientTape(persistent=True) as tape:
                    action, action_prob, state_value, prob_distribution = self._predict(state)
                    state_, reward, done = self.env.step(action)

                    reward  =  tf.convert_to_tensor(reward, dtype=tf.float32)
                    entropy = -tf.reduce_sum(prob_distribution * tf.math.log(prob_distribution + 1e-10))

                    delta = reward - state_value
                    if not done:
                        _, _, state_value_, _ = self._predict(state_)
                        delta += tf.multiply(state_value_, self.gamma)
                    log_action_prob = tf.math.log(action_prob)

                    bonus = tf.multiply(entropy, self.entropy_coeff)

                    actor_loss  = -tf.stop_gradient(delta) * log_action_prob - bonus
                    critic_loss = -tf.stop_gradient(delta) * state_value
                    total_loss  = actor_loss + critic_loss

                    state = state_

                grads_shared = tape.gradient(total_loss, self.actor_critic.shared_vars)
                grads_actor  = tape.gradient(total_loss, self.actor_critic.actor_vars)
                grads_critic = tape.gradient(total_loss, self.actor_critic.critic_vars)

                del tape

                # Apply gradients
                self.optimizer_shared.apply_gradients(zip(grads_shared, self.actor_critic.shared_vars))
                self.optimizer_actor.apply_gradients(zip(grads_actor, self.actor_critic.actor_vars))
                self.optimizer_critic.apply_gradients(zip(grads_critic, self.actor_critic.critic_vars))
            
            reward_end_of_episode.append(reward.numpy())
        
        plot_time_series(reward_end_of_episode)


if __name__ == "__main__":
    env = GcsimEnv(debug=True)
    agent = Agent(env, gamma=0.95)
    agent.learn(1000)
    env.close()