from gymnasium import spaces
from abc import abstractmethod
from typing import override
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
    
    def __init__(self, action_mapping: dict, debug: bool=False, debug_period: int=10) -> None:
        self.action_mapping = action_mapping
        
        self.debug = debug
        self.debug_period = debug_period
        self.episode_count = 0
        
        self.n_actions = self.get_n_actions()
        
        self.config_file = open(self.CONFIG_FILE_PATH, "r+")
        with open(self.CONFIG_HEADER_FILE_PATH, "r") as config_header_file:
            self.config_header_content = config_header_file.read()
        
        self.config_file_write_position = None

    @abstractmethod
    def reset(self) -> np.ndarray:        
        self.episode_count += 1
        self._reset_config_file()

    @abstractmethod
    def step(self, action) -> tuple[np.ndarray, float, bool]:
        pass

    def close(self) -> None:
        """
        Clean up resources when the environment is closed.
        """
        if self.config_file:
            self.config_file.close()

    def get_n_actions(self) -> int:
        return len(self.action_mapping)
    
    @abstractmethod
    def get_state_dim(self) -> int:
        pass

    def _analyze_gcsim_out(self, file_path: str) -> float:
        cd_duration, rps = 0, 0
        
        with open(file_path, "r") as gcsim_out:
            data = json.load(gcsim_out)
            rps = data["statistics"]["rps"]["mean"]
            
            failed_actions = data["statistics"]["failed_actions"]
            for failed_action in failed_actions:
                cd_duration += failed_action["skill_cd"]["mean"]
        
        return cd_duration, rps

    def _run_gcsim(self) -> float:
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

    def _update_config_file(self, action: int) -> None:
        self.config_file.seek(0, 2)
        self.config_file.write(self.action_mapping[action])
        self.config_file.flush()

    def _reset_config_file(self) -> None:
        if self.config_file_write_position is None:
            self.config_file.seek(0)
            self.config_file.write(self.config_header_content)
            self.config_file_write_position = self.config_file.tell()
            
        self.config_file.seek(self.config_file_write_position)
        self.config_file.truncate()


class GcsimV1(GcsimEnv):
    """
    GcsimEnv with fixed number of steps per episode
    """

    def __init__(self, action_mapping: dict, steps_per_episode=30, debug: bool=False, debug_period: int=10):
        super().__init__(action_mapping, debug, debug_period)

        self.steps_per_episode = steps_per_episode
        self.state = np.zeros((self.steps_per_episode + 1,), dtype=np.int32)
        self.state[0] = self.n_actions + 1 # the <start> token
        self.step_count = 0
    
    @override
    def reset(self) -> np.ndarray:
        """
        Reset the environment to an initial state and return the initial observation.
        """
        super().reset()

        self.step_count = 0
        self.state[1:] = 0

        return self.state
    
    @override
    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        """
        Execute one time step within the environment.
        """
        super().step(action)
        
        self.step_count += 1
        self.state[self.step_count] = action
        done = self.step_count == self.steps_per_episode

        reward = 0.0
        self._update_config_file(action)
        if done:
            raw_dps = self._run_gcsim()
            reward = self._compute_reward(raw_dps, self.GCSIM_OUT_FILE_PATH)
        
        return self.state, reward, done
    
    def _compute_reward(self, raw_dps: int, gcsim_out_file_path: str) -> float:
        raw_reward = raw_dps
        normalized_reward = raw_reward / 10000

        cd_duration, rps = self._analyze_gcsim_out(gcsim_out_file_path)
        penalty = cd_duration / 100
        
        return normalized_reward - penalty
    
    @override
    def get_state_dim(self) -> int:
        return self.steps_per_episode + 1

if __name__ == "__main__":
    action_mapping = {
        1: "alhaitham attack;",
        2: "alhaitham skill;",
        3: "furina skill;",
        4: "kuki skill;",
    }
    
    env = GcsimV1(action_mapping)

    state = env.reset()
    state_, reward, done = env.step(4)
    print(reward)
    
    env.close()