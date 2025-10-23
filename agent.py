import tensorflow as tf
import keras
import numpy as np
from actor_critic import ActorCritic
from env import GcsimEnv
from tqdm import tqdm
from custom_plot import plot_time_series

class Agent:
    def __init__(self, env: GcsimEnv, gamma=0.9, alpha=7e-4, entropy_coeff = 0.01, critic_loss_coeff = 0.5):
        self.env = env
        self.gamma = tf.constant(gamma, dtype=tf.float32)
        self.alpha = tf.constant(alpha, dtype=tf.float32)
        self.entropy_coeff = tf.constant(entropy_coeff, dtype=tf.float32)
        self.critic_loss_coeff = tf.constant(critic_loss_coeff, dtype=tf.float32)
        
        self.state_size = self.env.get_state_size()
        self.n_actions = self.env.get_n_actions()
        self.actor_critic = ActorCritic(self.state_size, self.n_actions)

        # self.cd_knowledge = np.array([], dtype=np.float32)
        # self.cd_knowledge = tf.convert_to_tensor([self.cd_knowledge], dtype=tf.float32)
        self.optimizer_shared = keras.optimizers.Adam(learning_rate=6e-4)
        self.optimizer_actor  = keras.optimizers.Adam(learning_rate=6e-4)
        self.optimizer_critic = keras.optimizers.Adam(learning_rate=1e-3)

    def predict(self, state):
        action, _, _ = self._predict(state)
        return action

    def _predict(self, state: np.ndarray):    
        state = tf.convert_to_tensor(state.reshape((1, -1)), dtype=tf.int32)
        prob_distribution, state_value = self.actor_critic(state)

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

            # all of these are list of tensors
            state_values = []
            log_action_probs = []
            entropies = []
            rewards = []
            discounted_returns = []
            
            with tf.GradientTape(persistent=True) as tape:
                while not done:
                    action, action_prob, state_value, prob_distribution = self._predict(state)
                    state_, reward, done = self.env.step(action)

                    log_action_prob = tf.math.log(action_prob)
                    entropy = -tf.reduce_sum(prob_distribution * tf.math.log(prob_distribution + 1e-10))
                    reward  =  tf.convert_to_tensor(reward, dtype=tf.float32)

                    state_values.append(state_value)
                    log_action_probs.append(log_action_prob)
                    entropies.append(entropy)
                    rewards.append(reward)

                    state = state_

                reward_end_of_episode.append(reward.numpy())

                G_t = tf.constant(0.0, dtype=tf.float32)
                for reward in reversed(rewards):
                    G_t = reward + self.gamma*G_t
                    discounted_returns.append(G_t)
                
                discounted_returns = list(reversed(discounted_returns))
                discounted_returns = tf.stack(discounted_returns)

                advantages = discounted_returns - tf.stack(state_values)

                # actor loss
                actor_loss_terms = [-tf.stop_gradient(adv) * log_action_prob for log_action_prob, adv in zip(log_action_probs, advantages)]
                entropy_bonus_terms = self.entropy_coeff * tf.stack(entropies)
                total_actor_loss = tf.reduce_sum(actor_loss_terms) - tf.reduce_sum(entropy_bonus_terms)

                # critic loss
                # critic_loss_terms = [-tf.stop_gradient(adv) * state_value for adv, state_value in zip(advantages, state_values)]
                # total_critic_loss = self.critic_loss_coeff * tf.reduce_sum(critic_loss_terms)
                total_critic_loss = tf.reduce_sum(tf.square(advantages)) # if full gradient descent
                
                total_loss = total_actor_loss + total_critic_loss

            # compute and apply gradients
            grads_shared = tape.gradient(total_loss, self.actor_critic.shared_vars)
            grads_actor  = tape.gradient(total_loss, self.actor_critic.actor_vars)
            grads_critic = tape.gradient(total_loss, self.actor_critic.critic_vars)
            del tape

            self.optimizer_shared.apply_gradients(zip(grads_shared, self.actor_critic.shared_vars))
            self.optimizer_actor.apply_gradients(zip(grads_actor, self.actor_critic.actor_vars))
            self.optimizer_critic.apply_gradients(zip(grads_critic, self.actor_critic.critic_vars))
        
        plot_time_series(reward_end_of_episode)


if __name__ == "__main__":
    env = GcsimEnv(debug=True, cd_penalty_factor=0.4, rps_reward_factor=0.05)
    agent = Agent(env, gamma=0.9, entropy_coeff=0.02, critic_loss_coeff=0.5)
    agent.learn(2000)        

    env.close()