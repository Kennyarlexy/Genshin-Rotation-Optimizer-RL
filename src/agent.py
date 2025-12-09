import tensorflow as tf
import keras
import numpy as np
from pathlib import Path
from actor_critic import ActorCritic
from env import GcsimEnv, GcsimV1, GcsimV2, GcsimState
from tqdm import tqdm
from custom_plot import plot_time_series

SCRIPT_PATH = Path(__file__)
PROJECT_ROOT = SCRIPT_PATH.parent.parent

class Agent:
    def __init__(self, env: GcsimEnv, gamma=1.0, alpha=6e-3, n_step = 1, entropy_coeff=0.01, critic_loss_coeff=0.5):
        self.env = env
        self.gamma = tf.constant(gamma, dtype=tf.float32)
        self.alpha = tf.constant(alpha, dtype=tf.float32)
        self.n_step = n_step
        self.entropy_coeff = tf.constant(entropy_coeff, dtype=tf.float32)
        self.critic_loss_coeff = tf.constant(critic_loss_coeff, dtype=tf.float32)
        
        self.seq_len = self.env.get_seq_len()
        self.n_actions = self.env.get_n_actions()

        n_special_actions = self.env.get_n_special_actions()
        self.actor_critic = ActorCritic(self.seq_len, self.n_actions, n_special_actions)
        self.optimizer = keras.optimizers.Adam(learning_rate=alpha)

        self.cumulative_reward_history = []
        self.n_loaded_episodes = 0

    def load(self, weights_h5_path, final_return_history_path):
        self._load_weights(weights_h5_path)
        self._load_final_return_history(final_return_history_path)

    def save(self, weights_h5_path, final_return_history_path):
        self._save_weights(weights_h5_path)
        self._save_final_return_history(final_return_history_path)

    def _load_weights(self, weights_h5_path):
        dummy_input = {
            "action_seq": tf.zeros((1, self.seq_len)), 
            "action_frames": tf.zeros((1, self.seq_len)), 
            "duration_left": tf.zeros((1, 1)),
        }

        self.actor_critic(dummy_input)
        self.actor_critic.load_weights(weights_h5_path)
        
    def _save_weights(self, weight_h5_path):
        self.actor_critic.save_weights(weight_h5_path)

    def _load_final_return_history(self, final_return_history_path):
        with open(final_return_history_path, 'r') as f:
            for value in f:
                value = float(value.strip())
                self.cumulative_reward_history.append(value)

        self.n_loaded_episodes = len(self.cumulative_reward_history)

    def _save_final_return_history(self, final_return_history_path):
        with open(final_return_history_path, 'w') as f:
            for value in self.cumulative_reward_history:
                value = value
                f.write(str(value) + "\n")
    
    def predict(self, state: GcsimState):
        action, _, _, _ = self._predict(state)
        return action

    def _predict(self, state: GcsimState):
        action_seq = tf.expand_dims(tf.convert_to_tensor(state.action_seq, dtype=tf.int32), axis=0)
        action_frames = None
        if state.action_frames is not None:
            action_frames = tf.expand_dims(tf.convert_to_tensor(state.action_frames, dtype=tf.float32), axis=0)
        duration_left = tf.convert_to_tensor([[state.duration_left or 0]], dtype=tf.float32)
        
        prob_distribution, state_value = self.actor_critic(
            {
                "action_seq": action_seq,
                "action_frames": action_frames,
                "duration_left": duration_left,
            }
        )
        prob_distribution = tf.squeeze(prob_distribution)
        state_value = tf.squeeze(state_value)

        sampled_action_index = np.random.choice(self.n_actions, p=prob_distribution.numpy())
        action = sampled_action_index

        action_prob = prob_distribution[sampled_action_index]

        return action, action_prob, state_value, prob_distribution

    def learn(self, n_episodes=1000):
        for episode in range(self.n_loaded_episodes + 1, self.n_loaded_episodes + n_episodes + 2):
            print("episode", episode)
            
            state = self.env.reset()
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
                    actions.append(action) 
                    rewards.append(reward)
                    states.append(state)

                    state = state_
                
                if done:
                    T = t1

                if t2 >= 1:
                    with tf.GradientTape() as tape:
                        _, _, state_value, prob_distribution = self._predict(states[t2-1])
                        if episode % 2 == 0:
                            print(prob_distribution)
                        else:
                            print(state_value)

                        action_prob = prob_distribution[actions[t2-1]]
                        log_action_prob = tf.math.log(action_prob + 1e-10)
                        entropy = -1 * tf.reduce_sum(prob_distribution * tf.math.log(prob_distribution + 1e-10)) * self.entropy_coeff
                        
                        t3 = min(T, t2 + self.n_step - 1) # point to last reward of the cumulative return
                        G_t = tf.constant(0, dtype=tf.float32)
                        if (t2 + self.n_step - 1 < T):
                            _, _, state_value_, _ = self._predict(states[(t2 + self.n_step - 1) - 1])
                            G_t = tf.stop_gradient(state_value_)
                        
                        for t in range(t3, t2-1, -1):
                            G_t = rewards[t-1] + self.gamma * G_t

                        advantage = tf.stop_gradient(G_t - state_value)

                        actor_loss  = -1 * advantage * log_action_prob 
                        # critic_loss = -1 * advantage * state_value * self.critic_loss_coeff
                        critic_loss = tf.square(G_t - state_value) * self.critic_loss_coeff
                        total_loss = actor_loss + critic_loss - entropy

                    # print("actor_loss =", actor_loss)
                    # print("critic_loss =", critic_loss)
                    # print("before =", state_value)

                    grads = tape.gradient(total_loss, self.actor_critic.trainable_variables)
                    self.optimizer.apply_gradients(zip(grads, self.actor_critic.trainable_variables))

                    # _, _, state_value_x, _ = self._predict(states[t2-1])
                    # print("after =", state_value_x)
                    # print("target =", G_t)
                    # print("difference =", state_value_x - state_value)
                    # print("------------------------------------")
                    
            self.cumulative_reward_history.append(cumulative_reward)
        
            if episode % 10 == 0:
                window = episode // 100 + 10
                plot_time_series(self.cumulative_reward_history, "reward_per_episode.png", window=window)
        
if __name__ == "__main__":
    WEIGHTS_H5_PATH = PROJECT_ROOT / 'models' / 'actor_critic.weights.h5'
    FINAL_RETURN_HISTORY_PATH = PROJECT_ROOT / 'var' / 'final_return_history.txt'

    action_list = ["alhaitham attack", "alhaitham skill", "furina skill", "kuki skill"]
    
    env = GcsimV2(action_list, debug=False)
    agent = Agent(env, gamma=1.0, entropy_coeff=0.1, critic_loss_coeff=0.5, alpha=1e-4, n_step=5)
    # agent.load(WEIGHTS_H5_PATH, FINAL_RETURN_HISTORY_PATH)
    
    agent.learn(500)
    agent.save(WEIGHTS_H5_PATH, FINAL_RETURN_HISTORY_PATH)

    env.close()