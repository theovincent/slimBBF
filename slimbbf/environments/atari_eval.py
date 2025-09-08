import ale_py
import gymnasium as gym
import numpy as np
import cv2
from typing import Tuple
import jax


class AtariEval:
    def __init__(self, name: str, sticky_actions: bool, n_envs: int) -> None:
        self.name = name
        self.n_envs = n_envs
        self.state_height, self.state_width = (84, 84)
        self.n_stacked_frames = 4
        self.n_skipped_frames = 4

        gym.register_envs(ale_py)

        # start with list of envs for NOOP initialization
        self.raw_envs = [
            gym.make(
                f"ALE/{self.name}-v5",
                full_action_space=False,
                frameskip=1,
                repeat_action_probability=0.25 if sticky_actions else 0.0,
                max_num_frames_per_episode=108_000,
                obs_type="grayscale",
            )
            for _ in range(n_envs)
        ]

        self.n_actions = self.raw_envs[0].action_space.n
        self.original_state_height, self.original_state_width = self.raw_envs[0].observation_space._shape
        self.screen_buffers = np.zeros(
            (n_envs, 2, self.original_state_height, self.original_state_width), dtype=np.uint8
        )
        self.states_ = np.zeros((n_envs, self.state_height, self.state_width, self.n_stacked_frames), dtype=np.uint8)
        self.n_lives = np.zeros(n_envs, dtype=np.int32)
        self.game_over_mask = np.zeros(n_envs, dtype=np.uint8)

    def reset_with_noop(self, key):
        for env_id in range(self.n_envs):
            key, env_id_key = jax.random.split(key)
            self.reset_with_noop_id(env_id, env_id_key)

        # Create async vectorized env for faster step(), starting from state after NOOP initialization
        self.envs = gym.vector.AsyncVectorEnv([lambda e=env: e for env in self.raw_envs])
        self.n_steps = 0

    def reset_with_noop_id(self, env_id, key):
        self.reset_id(env_id)
        n_noops = jax.random.randint(key, (), 0, 30)  # max_noops = 30
        for _ in range(n_noops):
            terminal = self.noop_step_id(env_id)
            if terminal:
                self.reset_id(env_id)

    def reset_id(self, env_id) -> None:
        obs_, info_ = self.raw_envs[env_id].reset()

        self.n_lives[env_id] = info_["lives"]  # to terminate on loss life

        self.screen_buffers[env_id, 0] = obs_
        self.screen_buffers[env_id, 1].fill(0)

        self.states_[env_id, :, :, -1] = self.resize(self.screen_buffers[env_id, 0])

    def noop_step_id(self, env_id):
        for idx_frame in range(self.n_skipped_frames):
            obs_, _, terminal_, _, info_ = self.raw_envs[env_id].step(0)  # action=0 is NOOP, ignore reward in this step

            # terminate on loss life
            terminal = terminal_ or (info_["lives"] < self.n_lives[env_id])

            if idx_frame >= self.n_skipped_frames - 2:
                self.screen_buffers[env_id, idx_frame - (self.n_skipped_frames - 2)] = obs_

            if terminal:
                break

        pooled = np.max(self.screen_buffers[env_id], axis=0)
        resized = self.resize(pooled)

        self.states_ = np.roll(self.states_, -1, axis=-1)
        self.states_[env_id, :, :, -1] = resized

        return terminal

    @property
    def states(self) -> np.ndarray:
        return np.array(self.states_, dtype=np.float32)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rewards = np.zeros(self.n_envs, dtype=np.float32)

        for idx_frame in range(self.n_skipped_frames):
            obs_, rewards_, game_over_, _, info_ = self.envs.step(actions)
            rewards += rewards_ * (1 - self.game_over_mask)
            self.game_over_mask = np.logical_or(self.game_over_mask, game_over_)

            # zero out state in envs where loss of life occurs
            self.states_[info_["lives"] < self.n_lives] = 0
            self.n_lives = info_["lives"]

            if idx_frame >= self.n_skipped_frames - 2:
                self.screen_buffers[:, idx_frame - (self.n_skipped_frames - 2)] = obs_

        pooled = np.max(self.screen_buffers, axis=1)
        resized = np.stack([self.resize(pooled[i]) for i in range(self.n_envs)], axis=0)

        self.states_ = np.roll(self.states_, -1, axis=-1)
        self.states_[:, :, :, -1] = resized

        self.n_steps += 1

        return rewards

    def resize(self, frame: np.ndarray) -> np.ndarray:
        return np.asarray(
            cv2.resize(frame, (self.state_width, self.state_height), interpolation=cv2.INTER_AREA),
            dtype=np.uint8,
        )
