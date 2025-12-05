from abc import abstractmethod
from typing import override
from pathlib import Path
import numpy as np
import subprocess
import re
import json

SCRIPT_PATH = Path(__file__)
PROJECT_ROOT = SCRIPT_PATH.parent.parent


class GcsimEnv:
    """
    A custom environment following the Gymnasium interface.
    """
    GCSIM_PATH = PROJECT_ROOT / "bin" / "gcsim.exe"
    GCSIM_OUT_FILE_PATH = PROJECT_ROOT / "gcsim_output" / "gcsim_out.json"
    CONFIG_FILE_PATH = PROJECT_ROOT / "gcsim_config" / "config.txt"
    CONFIG_HEADER_FILE_PATH = PROJECT_ROOT / "gcsim_config" / "config_header.txt"
    
    
    DEFAULT_OPTIONS = {
        "iteration": 10,
        "swap_delay": 14,
    }

    DEFAULT_TARGET = {
        "lvl": 100,
        "type": "dummy",
        "resist": 0.1,
        "particle_threshold": 520000,
        "particle_drop_count": 3,
    }
    
    def __init__(self, action_mapping: dict, debug: bool=False, debug_period: int=10, options: dict | None=None, target: dict | None=None) -> None:
        self.debug = debug
        self.debug_period = debug_period

        self.action_mapping = action_mapping
        self.episode_count = 0
        self.n_actions = self.get_n_actions()
        
        self.config_file = open(self.CONFIG_FILE_PATH, "r+")
        with open(self.CONFIG_HEADER_FILE_PATH, "r") as config_header_file:
            self.config_header_content = config_header_file.read()
        
        self.options = options or self.DEFAULT_OPTIONS
        self.enemy = target or self.DEFAULT_TARGET
        self.config_file_write_position = None

    @abstractmethod
    def reset(self) -> np.ndarray:
        """
        Reset the environment to an initial state and return the initial observation.
        """
        
        self.episode_count += 1
        self._reset_config_file()

    @abstractmethod
    def step(self, action) -> tuple[np.ndarray, float, bool]:
        """
        Execute one time step within the environment.
        """
        pass

    def close(self) -> None:
        """
        Clean up resources when the environment is closed
        """
        if self.config_file:
            self.config_file.close()

    def get_n_actions(self) -> int:
        return len(self.action_mapping)
    
    @abstractmethod
    def get_state_dim(self) -> int:
        pass

    def _analyze_gcsim_out(self) -> float:
        cd_duration, rps = 0, 0
        
        with open(self.GCSIM_OUT_FILE_PATH, "r") as gcsim_out:
            data = json.load(gcsim_out)
            rps = data["statistics"]["rps"]["mean"]
            
            failed_actions = data["statistics"]["failed_actions"]
            for failed_action in failed_actions:
                cd_duration += failed_action["skill_cd"]["mean"]
        
        return cd_duration, rps

    def _run_gcsim(self) -> tuple[float, float, float]:
        """
        return a tuple of dmg, duration, dps
        """
        
        try:
            result = subprocess.run([self.GCSIM_PATH, '-c', self.CONFIG_FILE_PATH, '-out', self.GCSIM_OUT_FILE_PATH], check=True, capture_output=True, text=True)
            output = result.stdout

            match_dmg = re.search(r"Average (\d+\.\d+) damage", output)
            dmg = float(match_dmg.group(1))

            match_duration = re.search(r"over (\d+\.\d+) seconds", output)
            duration = float(match_duration.group(1))

            match_dps = re.search(r"in (\d+) dps", output)
            dps = float(match_dps.group(1))

            if self.debug is True and self.episode_count % self.debug_period == 0:
                print(output)
            
            return dmg, duration, dps

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
            options = self._stringify_parameters(self.options, "options")
            target  = self._stringify_parameters(self.enemy, "target")
            self.config_file.write(self.config_header_content + options + target)
            self.config_file_write_position = self.config_file.tell()
            
        self.config_file.seek(self.config_file_write_position)
        self.config_file.truncate()

    def _stringify_parameters(self, params: dict, name: str) -> str:
        result = name
        for k, v in params.items():
            result += (" " + k + "=" + str(v))
        
        result += ";\n"
        return result


class GcsimV1(GcsimEnv):
    """
    GcsimEnv with fixed number of steps per episode
    """

    def __init__(self, action_mapping: dict, steps_per_episode=30, debug: bool=False, debug_period: int=10, options: dict | None=None, target: dict | None=None):
        super().__init__(action_mapping, debug, debug_period, options, target)

        self.steps_per_episode = steps_per_episode
        self.state = np.zeros((self.steps_per_episode + 1,), dtype=np.int32)
        self.state[0] = self.n_actions + 1 # the <start> token
        self.step_count = 0
    
    @override
    def reset(self) -> np.ndarray:
        super().reset()

        self.step_count = 0
        self.state[1:] = 0

        return self.state
        
    def _compute_reward(self, dps: float) -> float:
        raw_reward = dps
        normalized_reward = raw_reward / 10000

        cd_duration, rps = self._analyze_gcsim_out()
        penalty = cd_duration / 100
        
        return normalized_reward - penalty
    
    @override
    def get_state_dim(self) -> int:
        return self.steps_per_episode + 1
    
    @override
    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        self.step_count += 1
        self.state[self.step_count] = action
        done = self.step_count == self.steps_per_episode

        reward = 0.0
        self._update_config_file(action)
        if done:
            _, _, dps = self._run_gcsim()
            reward = self._compute_reward(dps)
        
        return self.state, reward, done


# class GcsimV2(GcsimEnv):
#     """
#     GcsimEnv with fixed duration in seconds
#     """
    
#     def __init__(self, action_mapping: dict, duration: float=30, debug: bool=False, debug_period: int=10):
#         super().__init__(action_mapping, debug, debug_period)
        
#         self.duration = duration
#         self.state = np.zeros((150,), dtype=np.int32)
#         self.state[0] = self.n_actions + 1 # the <start> token
#         self.step_count = 0

#     @override
#     def reset(self) -> np.ndarray:
#         super().reset()

#         self.step_count = 0
#         self.state[1:] = 0

#         return self.state
        
#     @override
#     def step(self, action: int) -> tuple[np.ndarray, float, bool]:
#         super().step(action)

#         dmg, duration, _ = self._run_gcsim()
        
#         self.step_count += 1
#         self.state[self.step_count] = action
#         done = duration == self.duration

#         reward = 0.0
#         self._update_config_file(action)
        
#         return self.state, reward, done

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
    env._run_gcsim()
    print(reward)
    
    env.close()