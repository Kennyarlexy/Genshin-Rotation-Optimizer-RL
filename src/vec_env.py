from env import GcsimState, GcsimEnv, GcsimV1, GcsimV2, GcsimRunInfo
from collections.abc import Iterable
import numpy as np

class SyncVectorGcsimEnv:
    def __init__(self, fn, n_envs: int=4):
        """
        fn: A function that creates environments. 
        """
        self.envs: list[GcsimEnv] = [fn() for _ in range(n_envs)]
        self.n_envs = n_envs
        
    def reset(self) -> list[GcsimState]:
        states = [env.reset() for env in self.envs]
        return states

    def step(self, actions: Iterable[int]) -> tuple[list[GcsimState], np.ndarray, np.ndarray, list[GcsimRunInfo | None]]:
        results = zip(*[env.step(action) for env, action in zip(self.envs, actions)])
        states, rewards, dones, run_infos = results

        return list(states), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.bool), list(run_infos)
    
    def close(self) -> None:
        for env in self.envs:
            env.close()
    

if __name__ == "__main__":
    action_list = ["alhaitham attack"]
    env = SyncVectorGcsimEnv(lambda: GcsimV2(action_list), n_envs=4)
    state = env.reset()
    print(state)
    env.close()