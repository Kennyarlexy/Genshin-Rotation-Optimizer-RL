import tensorflow as tf
import tensorflow_probability as tfp
import keras
import numpy as np
import traceback
from typing import Any
from custom_data_structure import FastDeque
from pathlib import Path
from actor_critic import ActorCritic
from env import GcsimEnv, GcsimV1, GcsimV2, GcsimState
from vec_env import SyncVectorGcsimEnv
from tqdm import tqdm
from custom_plot import plot_time_series

SCRIPT_PATH = Path(__file__)
PROJECT_ROOT = SCRIPT_PATH.parent.parent

class Agent:
    def __init__(self, env: SyncVectorGcsimEnv, gamma=1.0, alpha=6e-3, n_step = 1, entropy_coeff=0.01, critic_loss_coeff=0.5):
        self.env = env
        self.gamma = tf.constant(gamma, dtype=tf.float32)
        self.alpha = tf.constant(alpha, dtype=tf.float32)
        self.n_step = n_step
        self.entropy_coeff = tf.constant(entropy_coeff, dtype=tf.float32)
        self.critic_loss_coeff = tf.constant(critic_loss_coeff, dtype=tf.float32)
        
        self.seq_len = self.env.envs[0].get_seq_len()
        self.n_actions = self.env.envs[0].get_n_actions()

        n_special_actions = self.env.envs[0].get_n_special_actions()
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
        action_prob_dist, _ = self._forward(state)
        return action_prob_dist.sample()

    def _forward(self, inputs: dict) -> tuple[Any, tf.Tensor]:
        action_prob_dist, state_value = self.actor_critic(inputs)
        action_prob_dist = tfp.distributions.Categorical(probs=action_prob_dist)

        return action_prob_dist, state_value
    
    def _unpack_state(self, state: GcsimState) -> dict:
        """
        Unpack into dictionary containing tensors with correct shapes for feed forward
        """

        action_seq = tf.convert_to_tensor(state.action_seq, dtype=tf.int32)

        action_frames = None
        if state.action_frames is not None:
            action_frames = tf.convert_to_tensor(state.action_frames, dtype=tf.float32)
        
        duration_left = None
        if state.duration_left is not None:
            duration_left = tf.expand_dims(tf.convert_to_tensor(state.duration_left, dtype=tf.float32), axis=-1)

        unpacked_state = {
            "action_seq": action_seq,
            "action_frames": action_frames,
            "duration_left": duration_left,
        }

        return unpacked_state

    def learn(self, steps=1000):
        states = FastDeque(self.n_step + 1)
        actions = FastDeque(self.n_step)
        rewards = FastDeque(self.n_step)
        dones = FastDeque(self.n_step)

        states.push_back(self.env.reset())
        # to learn for "steps" number of times, we need steps + self.n_step - 1 interactions
        for step in range(1, steps + self.n_step):
            action_prob_dist, _ = self._forward(self._unpack_state(states[-1]))

            action = action_prob_dist.sample()
            state_, reward, done = self.env.step(action)
            states.push_back(state_)
            actions.push_back(action)
            rewards.push_back(reward)
            dones.push_back(done)
            
            if step >= self.n_step:
                state: GcsimState = states.pop_front()
                action: tf.Tensor = actions.pop_front()
                
                _, state_value_ = self._forward(self._unpack_state(states[-1]))
                G_t = tf.stop_gradient(tf.squeeze(state_value_, axis=-1))
                for t in reversed(range(self.n_step)):
                    G_t = rewards[t] + self.gamma * G_t * (1 - dones[t])

                rewards.pop_front()
                dones.pop_front()

                print(f"step {step - self.n_step + 1}  |  ", end="")
                self._update_network(self._unpack_state(state), action, G_t)

    @tf.function
    def _update_network(self, unpacked_state, action, G_t):
        with tf.GradientTape() as tape:
            action_prob_dist, state_value = self._forward(unpacked_state)

            log_prob = action_prob_dist.log_prob(action)

            state_value = tf.squeeze(state_value, axis=-1)
            advantage   = tf.stop_gradient(G_t - state_value)
            entropy     = tf.reduce_sum(action_prob_dist.entropy())
            actor_loss  = -1 * tf.reduce_sum(advantage * log_prob)
            critic_loss = tf.reduce_sum(tf.square(G_t - state_value))
            total_loss  = (actor_loss + self.critic_loss_coeff * critic_loss - self.entropy_coeff * entropy) / self.env.n_envs

            tf.print("loss", total_loss)
        grads = tape.gradient(total_loss, self.actor_critic.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.actor_critic.trainable_variables))

        
if __name__ == "__main__":
    WEIGHTS_H5_PATH = PROJECT_ROOT / 'models' / 'actor_critic.weights.h5'
    FINAL_RETURN_HISTORY_PATH = PROJECT_ROOT / 'var' / 'final_return_history.txt'

    action_list = ["alhaitham skill", "alhaitham attack", "furina skill", "kuki skill"]
    
    env = SyncVectorGcsimEnv(lambda: GcsimV2(action_list, debug=False, auto_reset=True), n_envs=6)
    
    try:
        agent = Agent(env, gamma=1.0, entropy_coeff=0.1, critic_loss_coeff=0.5, alpha=5e-5, n_step=10)
        agent.load(WEIGHTS_H5_PATH, FINAL_RETURN_HISTORY_PATH)
        agent.learn(100)
        agent.save(WEIGHTS_H5_PATH, FINAL_RETURN_HISTORY_PATH)
    except:
        traceback.print_exc() 

    env.close()
    print("Training finished")