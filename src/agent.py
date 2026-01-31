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


SCRIPT_PATH = Path(__file__)
PROJECT_ROOT = SCRIPT_PATH.parent.parent


class Agent:
    def __init__(self, train_env: SyncVectorGcsimEnv | None=None, eval_env: SyncVectorGcsimEnv | None=None, gamma=1.0, alpha=6e-3, t_max = 1, entropy_coeff=0.01, critic_loss_coeff=0.5, eval_freq: int=100, n_eval_episodes: int=10):
        self.train_env = train_env
        self.eval_env = eval_env
        self.gamma = tf.constant(gamma, dtype=tf.float32)
        self.alpha = tf.constant(alpha, dtype=tf.float32)
        self.t_max = t_max
        self.entropy_coeff = tf.constant(entropy_coeff, dtype=tf.float32)
        self.critic_loss_coeff = tf.constant(critic_loss_coeff, dtype=tf.float32)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        
        self.seq_len = self.train_env.envs[0].get_seq_len()
        self.n_actions = self.train_env.envs[0].get_n_actions()

        n_special_actions = self.train_env.envs[0].get_n_special_actions()
        self.actor_critic = ActorCritic(self.seq_len, self.n_actions, n_special_actions)
        self.optimizer = keras.optimizers.AdamW(learning_rate=alpha)

        self.cumulative_reward_history = []
        self.n_loaded_episodes = 0

        self._build_network()

    def load(self, weights_h5_path, final_return_history_path):
        self.actor_critic.load_weights(weights_h5_path)
        self._load_final_return_history(final_return_history_path)

    def save(self, weights_h5_path, final_return_history_path):
        self._save_weights(weights_h5_path)
        self._save_final_return_history(final_return_history_path)

    def _build_network(self):
        dummy_input = {
            "action_seq": tf.zeros((1, self.seq_len), dtype=tf.int32), 
            "action_frames": tf.zeros((1, self.seq_len)), 
            "relative_action_frames": tf.zeros((1, self.seq_len)), 
            "duration_left": tf.zeros((1, 1)),
            "remaining_skill_cds": tf.zeros((1, 4)),
        }

        self.actor_critic(dummy_input)
        
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
    
    def select_action(self, state: GcsimState, greedy: bool=False):
        inputs = self._unpack_state(state)
        action_prob_dist, _ = self._forward(inputs)
        if greedy:
            return action_prob_dist.mode()
        
        return action_prob_dist.sample()

    @tf.function
    def _forward(self, inputs: dict) -> tuple[Any, tf.Tensor]:
        action_prob_dist, state_value = self.actor_critic(inputs)
        action_prob_dist = tfp.distributions.Categorical(probs=action_prob_dist)

        return action_prob_dist, state_value
    
    def _unpack_state(self, states: list[GcsimState]) -> dict:
        """
        Unpack into dictionary containing tensors with correct shapes for feed forward
        """

        action_seq = tf.convert_to_tensor([state.ctx_action_seq for state in states], dtype=tf.int32)

        action_frames = None
        if (action_frames_norm := states[0].ctx_action_frames_norm) is not None:
            action_frames = tf.convert_to_tensor([action_frames_norm] + [state.ctx_action_frames_norm for state in states[1:]], dtype=tf.float32)

        relative_action_frames = None
        if (relative_action_frames_norm := states[0].ctx_relative_action_frames_norm) is not None:
            relative_action_frames = tf.convert_to_tensor([relative_action_frames_norm] + [state.ctx_relative_action_frames_norm for state in states[1:]], dtype=tf.float32)
        
        duration_left = None
        if (duration_left_norm := states[0].duration_left_norm) is not None:
            duration_left = tf.expand_dims([duration_left_norm] + [state.duration_left_norm for state in states[1:]], axis=-1)

        rem_skill_cds = None
        if (rem_skill_cds_ratio := states[0].rem_skill_cds_ratio) is not None:
            rem_skill_cds = tf.convert_to_tensor([rem_skill_cds_ratio] + [state.rem_skill_cds_ratio for state in states[1:]], dtype=tf.float32)

        unpacked_state = {
            "action_seq": action_seq,
            "action_frames": action_frames,
            "relative_action_frames": relative_action_frames,
            "duration_left": duration_left,
            "remaining_skill_cds": rem_skill_cds,
        }

        return unpacked_state

    def learn(self, steps=1000, evaluate=False):
        if self.train_env is None:
            raise Exception("No environment to learn from, pass in train_env when constructing agent to learn.")
        
        states_buff = FastDeque(self.t_max + 1)
        actions_buff = FastDeque(self.t_max)
        rewards_buff = FastDeque(self.t_max)
        dones_buff = FastDeque(self.t_max)

        states_buff.push_back(self.train_env.reset())
        for step in range(1, steps + 1):
            action_prob_dists, _ = self._forward(self._unpack_state(states_buff[-1]))

            actions = action_prob_dists.sample()
            states_, rewards, dones, _ = self.train_env.step(actions)
            states_buff.push_back(states_)
            actions_buff.push_back(actions)
            rewards_buff.push_back(rewards)
            dones_buff.push_back(dones)
            
            if step % self.t_max == 0 or step == steps:
                _, state_values_ = self._forward(self._unpack_state(states_buff[-1]))

                G = tf.stop_gradient(tf.squeeze(state_values_, axis=-1))
                n_step_max = min(self.t_max, rewards_buff.size)
                Gs = [None] * n_step_max
                for t in reversed(range(n_step_max)):
                    rewards: np.ndarray = rewards_buff.pop_back()
                    dones: np.ndarray = dones_buff.pop_back()
                    G = rewards + self.gamma * G * (1 - dones)
                    Gs[t] = G
                
                accum_grads = [tf.zeros_like(var) for var in self.actor_critic.trainable_variables]
                for t in range(n_step_max):
                    states: GcsimState = states_buff.pop_front()
                    actions: tf.Tensor = actions_buff.pop_front()
                
                    print(f"step {step - n_step_max + 1 + t}  |  ", end="")
                    grads = self._compute_grads(self._unpack_state(states), actions, Gs[t])
                    accum_grads = self._add_grads(accum_grads, grads)
                
                self.optimizer.apply_gradients(zip(accum_grads, self.actor_critic.trainable_variables))

            if evaluate and step % self.eval_freq == 0:
                print("Paused for evaluation...")
                self.evaluate()
                self.evaluate(greedy=True)

    @tf.function
    def _compute_grads(self, unpacked_state, actions, Gs):
        with tf.GradientTape() as tape:
            action_prob_dists, state_values = self._forward(unpacked_state)

            log_probs = action_prob_dists.log_prob(actions)

            state_values = tf.squeeze(state_values, axis=-1)
            advantages   = tf.stop_gradient(Gs - state_values)
            entropy      = tf.reduce_sum(action_prob_dists.entropy())
            actor_loss   = -1 * tf.reduce_sum(advantages * log_probs)
            critic_loss  = tf.reduce_sum(tf.square(Gs - state_values))
            total_loss   = (actor_loss + self.critic_loss_coeff * critic_loss - self.entropy_coeff * entropy) / self.train_env.n_envs

            tf.print("loss", total_loss, " |  probs[0]", action_prob_dists.probs[0], " |  values[0]", state_values[0])

        grads = tape.gradient(total_loss, self.actor_critic.trainable_variables)
        return grads

    def _add_grads(self, grads_1, grads_2) -> list[tf.Tensor]:
        for i in range(len(grads_1)):
            grads_1[i] = grads_1[i] + grads_2[i]

        return grads_1

    def evaluate(self, greedy=False):
        if self.eval_env is None:
            raise Exception("No environment to evaluate from, pass in eval_env when constructing agent to evaluate.")
        
        if self.eval_env.n_envs > 1:
            raise Exception("Can only evaluate on single instance of environment when parallelized.")
        
        total_reward, total_dmg = 0, 0
        for _ in range(self.n_eval_episodes):
            state = self.eval_env.reset()
            done = False
            while not done:
                action = self.select_action(state, greedy=greedy)
                state, _reward, _done, info = self.eval_env.step(action)
                total_reward += np.squeeze(_reward)
                done = np.squeeze(_done)
            
            total_dmg += info[0].total_dmg

        avg_reward = total_reward / self.n_eval_episodes
        avg_dmg = total_dmg / self.n_eval_episodes

        print(f"average reward for {self.n_eval_episodes} episodes run{' (greedy)' if greedy else ''}: {avg_reward}")
        print(f"average damage: {avg_dmg}")
        
        return avg_reward

        
if __name__ == "__main__":
    WEIGHTS_H5_PATH = PROJECT_ROOT / 'models' / 'actor_critic.weights.h5'
    FINAL_RETURN_HISTORY_PATH = PROJECT_ROOT / 'var' / 'final_return_history.txt'

    action_list = ["alhaitham skill", "alhaitham attack:2", "furina skill", "kuki skill"]
    
    train_env = SyncVectorGcsimEnv(lambda: GcsimV2(action_list, auto_reset=True), n_envs=6)
    eval_env  = SyncVectorGcsimEnv(lambda: GcsimV2(action_list, auto_reset=False), n_envs=1)
    
    try:
        agent = Agent(
            train_env, eval_env, 
            gamma=0.98, 
            entropy_coeff=0.030,
            critic_loss_coeff=0.5, 
            alpha=1e-4, 
            t_max=6,
            eval_freq=750,
        )
        agent.load(WEIGHTS_H5_PATH, FINAL_RETURN_HISTORY_PATH)
        agent.learn(2000, evaluate=True)
        agent.save(WEIGHTS_H5_PATH, FINAL_RETURN_HISTORY_PATH)
    except:
        traceback.print_exc() 

    train_env.close()
    print("Training finished")