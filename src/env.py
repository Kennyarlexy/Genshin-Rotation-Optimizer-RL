from abc import ABC, abstractmethod
from typing import override
from pathlib import Path
from dataclasses import dataclass, field, InitVar
from utils import return_none_on_error
import numpy as np
import subprocess
import re
import json
import copy

SCRIPT_PATH = Path(__file__)
PROJECT_ROOT = SCRIPT_PATH.parent.parent

@dataclass
class GcsimState:
    """
    :var action_seq: sequence of action tokens encoded in int
    :var action_frames: frames when an action happen corresponding to action_seq
    :var duration_left: how many seconds left until termination (running out of time)
    :var rem_skill_cds_norm: ratio between current cds (affected by cd reductions) and the original cooldown

    TODO:
    docs for the rest of the attrs
    """
    action_seq: np.ndarray
    action_frames: np.ndarray | None = None
    duration_left: float | None = None
    rem_skill_cds_ratio: np.ndarray | None = None

    # special/private attributes
    duration: InitVar[float | None] = None
    _duration: float | None = field(init=False, repr=False)
    seq_ctx_window: InitVar[float | None] = None
    _seq_ctx_window: float | None = field(init=False, repr=False)

    def __post_init__(self, duration, seq_ctx_window):
        self._duration = duration
        self._seq_ctx_window = seq_ctx_window
    
    @property
    @return_none_on_error
    def relative_action_frames(self) -> np.ndarray:
        result = self.action_frames
        mx = np.argmax(result)
        return np.where(result != GcsimEnv.UNDEFINED_VALUE, result[mx] - result, result)
    
    @property
    @return_none_on_error
    def relative_action_frames_norm(self) -> np.ndarray:
        result = self.relative_action_frames
        return np.where(result != GcsimEnv.UNDEFINED_VALUE, result / (60*self._duration), result)

    @property
    @return_none_on_error
    def action_frames_norm(self) -> np.ndarray:
        result = self.action_frames
        return np.where(result != GcsimEnv.UNDEFINED_VALUE, result / (60*self._duration), result)

    @property
    @return_none_on_error
    def duration_left_norm(self) -> float:
        return self.duration_left / self._duration
        
    @property
    @return_none_on_error
    def ctx_action_seq(self) -> np.ndarray:        
        mx = np.argmax(self.action_frames)
        outdated_cnt = np.sum((self.action_frames[mx] - self.action_frames[:mx+1]) > (self._seq_ctx_window*60))
        result = np.full_like(self.action_seq, GcsimEnv.SPECIAL_ACTIONS["<none>"])
        result[:mx-outdated_cnt+1] = self.action_seq[outdated_cnt:mx+1]

        return result
    
    @property
    @return_none_on_error
    def ctx_action_frames(self) -> np.ndarray:
        mx = np.argmax(self.action_frames)
        outdated_cnt = np.sum((self.action_frames[mx] - self.action_frames[:mx+1]) > (self._seq_ctx_window*60))
        result = np.full_like(self.action_frames, GcsimEnv.UNDEFINED_VALUE)
        result[:mx-outdated_cnt+1] = self.action_frames[outdated_cnt:mx+1]

        return result
    
    @property
    @return_none_on_error
    def ctx_action_frames_norm(self) -> np.ndarray:
        result = self.ctx_action_frames
        return np.where(result != GcsimEnv.UNDEFINED_VALUE, result / (60*self._duration), result)
    
    @property
    @return_none_on_error
    def ctx_relative_action_frames(self) -> np.ndarray:
        result = self.ctx_action_frames
        mx = np.argmax(result)
        return np.where(result != GcsimEnv.UNDEFINED_VALUE, result[mx] - result, result)
    
    @property
    @return_none_on_error
    def ctx_relative_action_frames_norm(self) -> np.ndarray:
        result = self.ctx_relative_action_frames
        return np.where(result != GcsimEnv.UNDEFINED_VALUE, result / (60*self._seq_ctx_window), result)


@dataclass
class GcsimRunInfo:
    total_dmg: float
    duration: float
    dps: float
    output: str


@dataclass
class GcsimSampleInfo:
    """
    :var action_frames: frames during which an action happen
    :var damages: accumulated damage in between each action
    :var skill_ready_frames: frame at which character at index i (provided by the sim) cooldown refreshes after accounting for all cooldown reductions
    :var skill_cd_durations: original cooldown of character at index i (provided by the sim) in frames before accounting for any cooldown reduction
    :var wasted_frames: number of frames the sim waited for each action to be available (due to cooldown etc)
    """
    action_frames: list[int]
    damages: list[float]
    skill_ready_frames: list[int]
    skill_cd_durations: list[int]
    wasted_frames: list[int]


class GcsimEnv(ABC):
    """
    A custom environment following the Gymnasium interface.
    """
    GCSIM_EXE_PATH = PROJECT_ROOT / "bin" / "gcsim.exe"

    # directory to store text files related to gcsim
    GCSIM_DIR_PATH = PROJECT_ROOT / "gcsim"

    RUN_DIR_PATH = GCSIM_DIR_PATH / "runs"
    
    CONFIG_FILE_BASENAME = "config.txt"
    OUT_FILE_BASENAME = "gcsim_out.json"
    SAMPLE_FILE_BASENAME = "gcsim_sample.json"

    CONFIG_HEADER_FILE_PATH = GCSIM_DIR_PATH / "config_header.txt"
    
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

    ACTION_TYPES = {"attack", "skill", "burst"}
    SPECIAL_ACTIONS = {
        "<none>": 0, 
        "<start>": 1,
    }

    N_CHARACTERS = 4

    UNDEFINED_VALUE = -1
    
    instance_cnt = 0
    
    def __init__(self, action_list: list[str], options: dict | None=None, target: dict | None=None, auto_reset: bool=False) -> None:
        self.done = False
        self.auto_reset = auto_reset

        self.action_list = action_list
        self.episode_count = 0
        self.n_actions = self.get_n_actions()
        self.n_special_actions = self.get_n_special_actions()
        
        self.seed = None
        self.options = options or self.DEFAULT_OPTIONS
        self.enemy = target or self.DEFAULT_TARGET
        self.config_file_write_position = None

        self.id = GcsimEnv.instance_cnt
        GcsimEnv.instance_cnt += 1
        self.config_file_path = self.RUN_DIR_PATH / str(self.id) / self.CONFIG_FILE_BASENAME
        self.gcsim_out_file_path = self.RUN_DIR_PATH  / str(self.id) /  self.OUT_FILE_BASENAME
        self.gcsim_sample_file_path = self.RUN_DIR_PATH  / str(self.id) /  self.SAMPLE_FILE_BASENAME
        
        self.config_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.gcsim_out_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.gcsim_sample_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.config_file_path.touch(exist_ok=True)
        self.config_file = open(self.config_file_path, "r+")
        with open(self.CONFIG_HEADER_FILE_PATH, "r") as config_header_file:
            self.config_header_content = config_header_file.read()

    @abstractmethod
    def reset(self) -> GcsimState:
        """
        Reset the environment to an initial state and return the initial observation.
        """
        
        self.done = False
        self.episode_count += 1
        self.seed = np.random.randint(100_000_000, 999_999_999)
        self._reset_config_file()

    @abstractmethod
    def step(self, action: int) -> tuple[GcsimState, float, bool, GcsimRunInfo | None]:
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

        print(f"env {self.id} closed...")

    def get_n_actions(self) -> int:
        return len(self.action_list)
    
    def get_n_special_actions(self) -> int:
        return len(self.SPECIAL_ACTIONS)
    
    @abstractmethod
    def get_seq_len(self) -> int:
        pass

    def _analyze_gcsim_out(self) -> tuple[float, float]:
        with open(self.gcsim_out_file_path, "r") as gcsim_out_file:
            data = json.load(gcsim_out_file)

        cd_duration, rps = 0, 0
        rps = data["statistics"]["rps"]["mean"]
        
        failed_actions = data["statistics"]["failed_actions"]
        for failed_action in failed_actions:
            cd_duration += failed_action["skill_cd"]["mean"]
        
        return cd_duration, rps
    
    def _analyze_gcsim_sample(self) -> GcsimSampleInfo:
        with open(self.gcsim_sample_file_path, "r") as gcsim_sample_file:
            data = json.load(gcsim_sample_file)

        action_frames = []
        damages = []
        skill_ready_frames = [0] * self.N_CHARACTERS
        skill_cd_durations = [-1] * self.N_CHARACTERS
        wasted_frames = [0]
        
        for log in data["logs"]:
            if is_real_action_event := (log["event"] == "action") and ("action" in log["logs"]) and (log["logs"]["action"] in self.ACTION_TYPES):
                action_frames.append(log["frame"])
                damages.append(0)
                wasted_frames.append(0)
            elif is_damage_event := (log["event"] == "damage") and ("self damage" not in log["logs"]["abil"]):
                damages[-1] += log["logs"]["damage"]
            elif is_cooldown_event := (log["event"] == "cooldown") and (log["msg"] == "skill cooldown triggered"):
                skill_ready_frames[log["char_index"]] = log["frame"] + log["logs"]["modified_cd_by_cdr"]
                skill_cd_durations[log["char_index"]] = log["logs"]["original_cd"]
            elif is_wasted_frame_event := (log["event"] == "sim") and log["msg"].endswith("action not ready"):
                wasted_frames[-1] += 1

        # wasted_frames[-1] is just there because it works by "looking forward" for a new action
        wasted_frames.pop()

        return GcsimSampleInfo(
            action_frames=action_frames,
            damages=damages,
            skill_ready_frames=skill_ready_frames,
            skill_cd_durations=skill_cd_durations,
            wasted_frames=wasted_frames,
        )

    def _run_gcsim(self) -> GcsimRunInfo:        
        result = subprocess.run(
            [self.GCSIM_EXE_PATH, '-c', self.config_file_path, '-out', self.gcsim_out_file_path, '-sample', self.gcsim_sample_file_path, '-seed', str(self.seed)], 
            check=True, 
            capture_output=True, 
            text=True
        )
        output = result.stdout

        match_dmg = re.search(r"Average (\d+\.\d+) damage", output)
        dmg = float(match_dmg.group(1))

        match_duration = re.search(r"over (\d+\.\d+) seconds", output)
        duration = float(match_duration.group(1))

        match_dps = re.search(r"in (\d+) dps", output)
        dps = float(match_dps.group(1))
        
        return GcsimRunInfo(
            total_dmg=dmg,
            duration=duration,
            dps=dps,
            output=output,
        )

    def _update_config_file(self, action: int) -> None:
        self.config_file.seek(0, 2)
        self.config_file.write(f"{self.action_list[action]};")
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
        parts = [f"{k}={v}" for k, v in params.items()]
        joined_params = " ".join(parts)
        
        return f"{name} {joined_params};\n"


class GcsimV1(GcsimEnv):
    """
    GcsimEnv with fixed number of steps per episode
    """

    def __init__(self, action_list: list[str], steps_per_episode=30, options: dict | None=None, target: dict | None=None, auto_reset: bool=False):
        super().__init__(action_list, options, target, auto_reset)

        self.steps_per_episode = steps_per_episode
        self.state = GcsimState(np.zeros((self.steps_per_episode + 1,), dtype=np.int32))
        self.state.action_seq[0] = self.SPECIAL_ACTIONS["<start>"]
        self.step_count = 0
    
    @override
    def reset(self) -> GcsimState:
        super().reset()

        self.step_count = 0
        self.state.action_seq[1:] = self.SPECIAL_ACTIONS["<none>"]

        return copy.deepcopy(self.state)
        
    def _compute_reward(self, dps: float) -> float:
        raw_reward = dps
        normalized_reward = raw_reward / 10000

        cd_duration, rps = self._analyze_gcsim_out()
        penalty = cd_duration / 100
        
        return normalized_reward - penalty
    
    @override
    def get_seq_len(self) -> int:
        return self.steps_per_episode + 1
    
    @override
    def step(self, action: int) -> tuple[GcsimState, float, bool, GcsimRunInfo | None]:
        if self.done and not self.auto_reset:
            return copy.deepcopy(self.state), 0.0, True, None
        
        self.step_count += 1
        self.state.action_seq[self.step_count] = action + self.n_special_actions
        done = self.step_count == self.steps_per_episode

        reward = 0.0
        self._update_config_file(action)
        run_info = None
        if done:
            run_info = self._run_gcsim()
            reward = self._compute_reward(run_info.dps)
            self.done = True
            if self.auto_reset:
                self.reset()
        
        return copy.deepcopy(self.state), reward, done, run_info


class GcsimV2(GcsimEnv):
    """
    GcsimEnv with fixed duration in seconds, default=60s

    Note:
    If you specify `'duration'` key in `options` dict, it will be ignored (replaced by the `duration` parameter).

    """

    DEFAULT_OPTIONS = {
        "iteration": 1,
        "swap_delay": 14,
    }

    DEFAULT_TARGET = {
        "lvl": 100,
        "resist": 0.1,
        "particle_threshold": 520000,
        "particle_drop_count": 3,
    }

    MAX_SEQ_LEN = 150 # this shouldn't be exceeded in any case
    
    def __init__(self, action_list: list[str], duration: float=60.0, options: dict | None=None, target: dict | None=None, auto_reset: bool=False):
        options = options or self.DEFAULT_OPTIONS
        target = target or self.DEFAULT_TARGET
        options["duration"] = duration

        super().__init__(action_list, options or self.DEFAULT_OPTIONS, target or self.DEFAULT_TARGET, auto_reset)
        
        self.duration = duration
        self.state = GcsimState(
            action_seq=np.zeros((self.MAX_SEQ_LEN,), dtype=np.int32), 
            action_frames=np.zeros((self.MAX_SEQ_LEN,), dtype=np.int32), 
            duration_left=self.duration,
            rem_skill_cds_ratio=np.zeros((self.N_CHARACTERS,), dtype=np.float32),
            duration=self.duration,
            seq_ctx_window=30.0,
        )
        self.state.action_frames[1:] = GcsimEnv.UNDEFINED_VALUE
        self.state.action_seq[0] = self.SPECIAL_ACTIONS["<start>"]
        self.step_count = 0

        self.last_action_frame = None

    @override
    def reset(self) -> GcsimEnv:
        super().reset()

        self.step_count = 0
        self.state.action_seq[1:] = self.SPECIAL_ACTIONS["<none>"]
        self.state.action_frames[1:] = GcsimEnv.UNDEFINED_VALUE
        self.state.duration_left = self.duration # normalized (full duration is 1)
        self.state.rem_skill_cds_ratio[:] = 0
        self.last_action_frame = None

        return copy.deepcopy(self.state)
        
    @override
    def step(self, action: int) -> tuple[GcsimState, float, bool, GcsimRunInfo | None]:        
        if self.done and not self.auto_reset:
            return copy.deepcopy(self.state), 0.0, True, None
        
        self._update_config_file(action)
        run_info = self._run_gcsim()
        sample_info = self._analyze_gcsim_sample()
        
        self.step_count += 1
        self.state.action_seq[self.step_count] = action + self.n_special_actions
        self.state.action_frames[self.step_count] = sample_info.action_frames[-1]
        self.state.duration_left = (self.duration - (sample_info.action_frames[-1] / 60))
        self.state.rem_skill_cds_ratio = self._compute_remaining_skill_cds(sample_info)

        reward = 0.0 
        done = (sample_info.action_frames[-1] == self.last_action_frame)
        self.last_action_frame = sample_info.action_frames[-1]

        assert abs(sum(sample_info.damages) - run_info.total_dmg) < 1.0, "damage from running the sim vs obtained from sample is different"
        if done:
            reward = sample_info.damages[-1]
            self.done = True
            if self.auto_reset:
                self.reset() # affect self.state under the hood
        elif self.step_count > 1:
            reward = sample_info.damages[-2]
        
        # get 1 reward for every 500K damage dealt between the last action and prev action
        reward /= 5e5
        # get 1 penalty for every 1200 wasted frames caused by the last action
        penalty = sample_info.wasted_frames[-1] / 1200

        reward -= penalty
        
        return copy.deepcopy(self.state), reward, done, run_info
    
    def _compute_remaining_skill_cds(self, sample_info: GcsimSampleInfo) -> np.ndarray:
        skill_ready_frames = np.array(sample_info.skill_ready_frames, dtype=np.float32)
        skill_cd_durations = np.array(sample_info.skill_cd_durations, dtype=np.float32)
        remaining_skill_cds = skill_ready_frames - sample_info.action_frames[-1]
        remaining_skill_cds[(remaining_skill_cds < 0) | (skill_cd_durations == -1)] = 0
        remaining_skill_cds /= skill_cd_durations
        
        return remaining_skill_cds

    @override
    def get_seq_len(self):
        return self.MAX_SEQ_LEN


if __name__ == "__main__":
    env_1 = GcsimV1(["alhaitham attack"])
    env_1.reset()
    env_1.step(0)
    env_1.close()

    env_2 = GcsimV2(["alhaitham attack"])
    env_2.reset()
    env_2.step(0)
    env_2.close()
