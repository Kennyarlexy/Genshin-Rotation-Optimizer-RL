import tensorflow as tf
import keras
import numpy as np
from actor_critic import ActorCritic
from env import GcsimEnv, GcsimV1
from tqdm import tqdm
from custom_plot import plot_time_series

class Agent:
    def __init__(self, env: GcsimEnv, gamma=1, alpha=6e-3, n_step = 1, entropy_coeff=0.01, critic_loss_coeff=0.5):
        self.env = env
        self.gamma = tf.constant(gamma, dtype=tf.float32)
        self.alpha = tf.constant(alpha, dtype=tf.float32)
        self.n_step = n_step
        self.entropy_coeff = tf.constant(entropy_coeff, dtype=tf.float32)
        self.critic_loss_coeff = tf.constant(critic_loss_coeff, dtype=tf.float32)
        
        self.state_dim = self.env.get_state_dim()
        self.n_actions = self.env.get_n_actions()

        self.actor_critic = ActorCritic(self.state_dim, self.n_actions)
        self.optimizer = keras.optimizers.Adam(learning_rate=alpha)

        self.cumulative_reward_history = []

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
                self.cumulative_reward_history.append(value)

    def _save_final_return_history(self, final_return_history_path):
        with open(final_return_history_path, 'w') as f:
            for value in self.cumulative_reward_history:
                value = value
                f.write(str(value) + "\n")
    
    def predict(self, state):
        action, _, _, _ = self._predict(state)
        return action

    def _predict(self, state: np.ndarray):    
        prob_distribution, state_value = self.actor_critic(state)

        sampled_action_index = np.random.choice(self.n_actions, p=prob_distribution.numpy()[0])
        action = sampled_action_index + 1 # used in env.step()

        action_prob = prob_distribution[0, sampled_action_index]
        state_value = state_value[0, 0]

        return action, action_prob, state_value, prob_distribution

    def learn(self, n_episodes=1000):
        for episode in range(1, n_episodes+1):
            print("episode", episode)
            
            state = tf.expand_dims(tf.convert_to_tensor(self.env.reset(), dtype=tf.int32), axis=0)
            done = False

            states = []
            rewards = []
            actions = []
            cumulative_reward = 0 # sum of raw rewards from beginning to end of episode
            
            T = 1000000000
            t1 = 0 # point to current interaction
            t2 = t1 - self.n_step + 1 # point to first reward of the cumulative return
            while t2 < T:
                t2 += 1
                if not done:
                    t1 += 1
                    action, _, _, _ = self._predict(state)
                    
                    state_, reward, done = self.env.step(action)
                    cumulative_reward += reward
                    reward = tf.convert_to_tensor(reward, dtype=tf.float32)
                    actions.append(action) 
                    rewards.append(reward)
                    states.append(state)

                    state = tf.expand_dims(tf.convert_to_tensor(state_, dtype=tf.int32), axis=0)
                
                if done:
                    T = t1

                if t2 >= 1:
                    with tf.GradientTape() as tape:
                        _, _, state_value, prob_distribution = self._predict(states[t2-1])
                        if episode % 2 == 0:
                            print(prob_distribution[0])
                        else:
                            print(state_value)

                        action_prob = prob_distribution[0, actions[t2-1]-1]
                        log_action_prob = tf.math.log(action_prob)
                        entropy = -tf.reduce_sum(prob_distribution * tf.math.log(prob_distribution + 1e-10))
                        
                        t3 = min(T, t2 + self.n_step - 1) # point to last reward of the cumulative return
                        G_t = tf.constant(0, dtype=tf.float32)
                        if (t2 + self.n_step - 1 < T):
                            _, _, state_value_, _ = self._predict(states[(t2 + self.n_step - 1) - 1])
                            G_t = state_value_
                        
                        for t in range(t3, t2-1, -1):
                            G_t = rewards[t-1] + self.gamma * G_t

                        advantage = G_t - state_value

                        actor_loss  = -1 * tf.stop_gradient(advantage) * log_action_prob
                        critic_loss = -1 * tf.stop_gradient(advantage) * state_value
                        total_loss = actor_loss + self.critic_loss_coeff * critic_loss - self.entropy_coeff * entropy

                    grads = tape.gradient(total_loss, self.actor_critic.trainable_variables)
                    self.optimizer.apply_gradients(zip(grads, self.actor_critic.trainable_variables))
                    
            self.cumulative_reward_history.append(cumulative_reward)
        
            if episode % 10 == 0:
                plot_time_series(self.cumulative_reward_history)
        
if __name__ == "__main__":
    WEIGHTS_H5_PATH = './actor_critic.weights.h5'
    FINAL_RETURN_HISTORY_PATH = './final_return_history.txt'

    action_mapping = {
        1: "alhaitham attack;",
        2: "alhaitham skill;",
        3: "furina skill;",
        4: "kuki skill;",
    }
    
    env = GcsimV1(action_mapping, debug=True)
    agent = Agent(env, gamma=1, entropy_coeff=0, critic_loss_coeff=2, alpha=3e-4, n_step=30)
    agent.load(WEIGHTS_H5_PATH, FINAL_RETURN_HISTORY_PATH)
    
    agent.learn(200)
    agent.save(WEIGHTS_H5_PATH, FINAL_RETURN_HISTORY_PATH)

    env.close()