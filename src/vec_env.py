from env import GcsimState, GcsimEnv, GcsimV1, GcsimV2
from collections.abc import Iterable
import numpy as np

class SyncVectorGcsimEnv:
    def __init__(self, fn, n_envs: int=4):
        """
        fn: A function that creates environments. 
        """
        self.envs: list[GcsimEnv] = [fn() for _ in range(n_envs)]
        self.n_envs = n_envs
        
    def reset(self) -> GcsimState:
        states = [env.reset() for env in self.envs]
        return self._stack_states(states)

    def step(self, actions: Iterable[int]) -> tuple[GcsimState, np.ndarray, np.ndarray]:
        results = zip(*[env.step(action) for env, action in zip(self.envs, actions)])
        states, rewards, dones = results
        
        return self._stack_states(states), np.array(rewards), np.array(dones)
    
    def close(self) -> None:
        for env in self.envs:
            env.close()

    def _stack_states(self, state_list: list[GcsimState]) -> GcsimState:        
        batch_action_seq = np.stack([s.action_seq for s in state_list])
        
        batch_action_frames = None
        if state_list[0].action_frames is not None:
            batch_action_frames = np.stack([s.action_frames for s in state_list])
        
        batch_relative_action_frames = None
        if state_list[0].relative_action_frames is not None:
            batch_relative_action_frames = np.stack([s.relative_action_frames for s in state_list])

        batch_duration_left = None
        if state_list[0].duration_left is not None:
            batch_duration_left = np.array([s.duration_left for s in state_list])

        return GcsimState(
            action_seq=batch_action_seq, 
            action_frames=batch_action_frames, 
            relative_action_frames=batch_relative_action_frames,
            duration_left=batch_duration_left
        )
    

if __name__ == "__main__":
    action_list = ["alhaitham attack"]
    env = SyncVectorGcsimEnv(lambda: GcsimV2(action_list))
    state = env.reset()
    print(state)