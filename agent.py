import tensorflow as tf
import keras
import numpy as np
from actor_critic import ActorCritic
from env import GcsimEnv
from tqdm import tqdm
from custom_plot import plot_time_series

class Agent:
    def __init__(self, env: GcsimEnv, gamma=1, alpha=6e-3, entropy_coeff = 0.01, critic_loss_coeff = 0.5, weights_h5_path = None, final_return_history_path = None):
        self.env = env
        self.gamma = tf.constant(gamma, dtype=tf.float32)
        self.alpha = tf.constant(alpha, dtype=tf.float32)
        self.entropy_coeff = tf.constant(entropy_coeff, dtype=tf.float32)
        self.critic_loss_coeff = tf.constant(critic_loss_coeff, dtype=tf.float32)
        
        self.state_dim = self.env.get_state_dim()
        self.n_actions = self.env.get_n_actions()

        self.actor_critic = ActorCritic(self.state_dim, self.n_actions)

        self.optimizer = keras.optimizers.Adam(learning_rate=alpha)

        self.final_return_history = []

    def load(self, weights_h5_path, final_return_history_path):
        self._load_weights(weights_h5_path)
        self._load_final_return_history(final_return_history_path)

    def save(self, weights_h5_path, final_return_history_path):
        self._save_weights(weights_h5_path)
        self._save_final_return_history(final_return_history_path)

    def _load_weights(self, weights_h5_path):
        dummy_input = tf.zeros((1, self.state_dim))
        self.actor_critic(dummy_input)
        self.actor_critic.load_weights(weights_h5_path)
        
    def _save_weights(self, weight_h5_path):
        self.actor_critic.save_weights(weight_h5_path)

    def _load_final_return_history(self, final_return_history_path):
        with open(final_return_history_path, 'r') as f:
            for value in f:
                value = float(value.strip())
                tensor = tf.convert_to_tensor(value, dtype=tf.float32)
                self.final_return_history.append(tensor)

    def _save_final_return_history(self, final_return_history_path):
        with open(final_return_history_path, 'w') as f:
            for value in self.final_return_history:
                value = value.numpy().item()
                f.write(str(value) + "\n")
    
    def predict(self, state):
        action, _, _ = self._predict(state)
        return action

    def _predict(self, state: np.ndarray):    
        prob_distribution, state_value = self.actor_critic(state)

        sampled_action_index = np.random.choice(self.n_actions, p=prob_distribution.numpy()[0])
        print(prob_distribution[0])
        action = sampled_action_index + 1 # used in env.step()

        action_prob = prob_distribution[0, sampled_action_index]
        state_value = state_value[0, 0]

        return action, action_prob, state_value, prob_distribution

    def learn(self, n_episodes=1000):
        for episode in range(1, n_episodes+1):
            print("episode", episode)
            
            state = tf.expand_dims(tf.convert_to_tensor(self.env.reset(), dtype=tf.int32), axis=0)
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

                    state = tf.expand_dims(tf.convert_to_tensor(state_, dtype=tf.int32), axis=0)

                G_t = tf.constant(0.0, dtype=tf.float32)
                for reward in reversed(rewards):
                    G_t = reward + self.gamma*G_t
                    discounted_returns.append(G_t)
                                
                self.final_return_history.append(G_t)

                discounted_returns = list(reversed(discounted_returns))
                discounted_returns = tf.stack(discounted_returns)

                advantages = discounted_returns - tf.stack(state_values)

                # actor loss
                actor_loss_terms = [-tf.stop_gradient(adv) * log_action_prob for log_action_prob, adv in zip(log_action_probs, advantages)]
                entropy_bonus_terms = self.entropy_coeff * tf.stack(entropies)
                total_actor_loss = tf.reduce_sum(actor_loss_terms) # - tf.reduce_sum(entropy_bonus_terms)

                # critic loss
                # critic_loss_terms = [tf.stop_gradient(adv) * state_value for adv, state_value in zip(advantages, state_values)]
                # total_critic_loss = self.critic_loss_coeff * tf.reduce_sum(critic_loss_terms) * tf.constant(-1, dtype=tf.float32)
                
                total_critic_loss = self.critic_loss_coeff * tf.reduce_sum(tf.square(advantages))
                
                total_loss = total_actor_loss + total_critic_loss
                print(total_actor_loss, total_critic_loss)

            # compute and apply gradients
            grads = tape.gradient(total_loss, self.actor_critic.trainable_variables)
            del tape
            self.optimizer.apply_gradients(zip(grads, self.actor_critic.trainable_variables))
        
        plot_time_series(self.final_return_history)
        
if __name__ == "__main__":
    WEIGHTS_H5_PATH = './actor_critic.weights.h5'
    FINAL_RETURN_HISTORY_PATH = './final_return_history.txt'
    
    env = GcsimEnv(debug=True, cd_penalty_factor=0.4, rps_reward_factor=0.05)
    agent = Agent(env, gamma=1, entropy_coeff=0.02, critic_loss_coeff=2, alpha=1e-3)
    agent.load(WEIGHTS_H5_PATH, FINAL_RETURN_HISTORY_PATH)
    print(agent.final_return_history)
    
    agent.learn(100)
    agent.save(WEIGHTS_H5_PATH, FINAL_RETURN_HISTORY_PATH)

    env.close()