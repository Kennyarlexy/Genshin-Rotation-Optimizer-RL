from gymnasium import spaces
import numpy as np
import subprocess
import re
import json

from normalizer import WelfordNormalizer


class GcsimEnv:
    """
    A custom environment following the Gymnasium interface.
    """
    GCSIM_PATH = "gcsim.exe"
    CONFIG_FILE_PATH = "config.txt"
    CONFIG_HEADER_FILE_PATH = "config_header.txt"
    GCSIM_OUT_FILE_PATH = "gcsim_out.json"

    ACTION_TO_STRING = {
        1: "alhaitham attack;",
        2: "alhaitham skill;",
        3: "furina skill;",
        4: "kuki skill;",
    }
    DEFAULT_EPISODE_LEN = 12
    
    def __init__(self, debug=False, debug_period=10, cd_penalty_factor=1, rps_reward_factor=1):
        self.debug = debug
        self.debug_period=debug_period
        self.episode_count = 0
        self.penalty_factor = cd_penalty_factor
        self.rps_reward_factor = rps_reward_factor
        
        n_actions = self.get_n_actions()

        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(
            low=0, 
            high=n_actions,
            shape=(GcsimEnv.DEFAULT_EPISODE_LEN,), 
            dtype=np.int32
        )
        
        self.max_episode_len = GcsimEnv.DEFAULT_EPISODE_LEN
        self.state = np.zeros((GcsimEnv.DEFAULT_EPISODE_LEN,), dtype=np.int32)
        self.move_count = 0
        self.config_file = open(GcsimEnv.CONFIG_FILE_PATH, "r+")
        with open(GcsimEnv.CONFIG_HEADER_FILE_PATH, "r") as config_header_file:
            self.config_header_content = config_header_file.read()
        
        self.config_file_write_position = None

        self.dps_normalizer = WelfordNormalizer()
        self.cd_normalizer = WelfordNormalizer()
        self.rps_normalizer = WelfordNormalizer()


    def reset(self, max_episode_len=None):
        """
        Reset the environment to an initial state and return the initial observation.
        """
        self.episode_count += 1
        self._reset_config_file()
        self.move_count = 0
        if max_episode_len:
            self.state = np.zeros((GcsimEnv.DEFAULT_EPISODE_LEN,), dtype=np.int32)
            self.max_episode_len=max_episode_len
        else:
            self.state[:] = 0

        return self.state

    def step(self, action: int):
        """
        Execute one time step within the environment.
        """        
        self.move_count += 1
        self.state[self.move_count-1] = action
        done = self.move_count == self.max_episode_len

        reward = 0
        self._update_config_file(action)
        if done:
            raw_dps = self._run_gcsim()
            reward = self._compute_reward(raw_dps, GcsimEnv.GCSIM_OUT_FILE_PATH)
        
        return self.state, reward, done

    def close(self):
        """
        Clean up resources when the environment is closed.
        """
        if self.config_file:
            self.config_file.close()

    def get_n_actions(self):
        return len(GcsimEnv.ACTION_TO_STRING)
    
    def get_state_size(self):
        return GcsimEnv.DEFAULT_EPISODE_LEN
    
    def _compute_reward(self, raw_dps: int, gcsim_out_file_path: str) -> float:
        normalized_dps = self.dps_normalizer.transform(raw_dps)
        
        cd_penalty, rps = self._analyze_gcsim_out(gcsim_out_file_path)
        # print(cd_penalty, rps)
        normalized_cd = self.cd_normalizer.transform(cd_penalty)
        normalized_rps = self.rps_normalizer.transform(rps)
        
        return normalized_dps - self.penalty_factor * normalized_cd + self.rps_reward_factor * normalized_rps
        # return normalized_dps - self.penalty_factor * cd_penalty + self.rps_reward_factor * rps

    def _analyze_gcsim_out(self, file_path: str) -> float:
        penalty, rps = 0, 0
        
        with open(file_path, "r") as gcsim_out:
            data = json.load(gcsim_out)
            rps = data["statistics"]["rps"]["mean"]
            
            failed_actions = data["statistics"]["failed_actions"]
            for failed_action in failed_actions:
                penalty += failed_action["skill_cd"]["mean"]
        
        return penalty, rps

    def _run_gcsim(self):
        try:
            result = subprocess.run([GcsimEnv.GCSIM_PATH, '-out', GcsimEnv.GCSIM_OUT_FILE_PATH], check=True, capture_output=True, text=True)
            output = result.stdout
            match  = re.search(r"in (\d+) dps", output)
            dps    = match.group(1)
            
            if self.debug is True and self.episode_count % self.debug_period == 0:
                print(output)
            
            return float(dps)

        except subprocess.CalledProcessError as e:
            print(f"Error: {e.stderr}")
            print(f"An unexpected error occurred: {e}")

    def _update_config_file(self, action: int):
        self.config_file.seek(0, 2)
        self.config_file.write(self.ACTION_TO_STRING[action])
        self.config_file.flush()

    def _reset_config_file(self):
        if self.config_file_write_position is None:
            self.config_file.seek(0)
            self.config_file.write(self.config_header_content)
            self.config_file_write_position = self.config_file.tell()
            
        self.config_file.seek(self.config_file_write_position)
        self.config_file.truncate()


if __name__ == "__main__":
    env = GcsimEnv()

    obs = env.reset()
    obs_, reward, done = env.step(4)
    print(reward)
    
    env.close()