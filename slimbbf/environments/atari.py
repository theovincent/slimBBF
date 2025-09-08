"""
The environment is inspired from https://github.com/google/dopamine/blob/master/dopamine/discrete_domains/atari_lib.py
"""

import ale_py
from typing import Tuple
import gymnasium as gym
import numpy as np
import jax
import jax.numpy as jnp
import cv2


class AtariEnv:
    def __init__(self, name: str, sticky_actions: bool) -> None:
        self.name = name
        self.state_height, self.state_width = (84, 84)
        self.n_stacked_frames = 4
        self.n_skipped_frames = 4

        gym.register_envs(ale_py)  # To use ale with gym which speeds up step()
        self.env = gym.make(
            f"ALE/{self.name}-v5",
            full_action_space=False,
            frameskip=1,
            repeat_action_probability=0.25 if sticky_actions else 0.0,
            max_num_frames_per_episode=108_000,
            obs_type="grayscale",
        ).env

        self.n_actions = self.env.action_space.n
        self.original_state_height, self.original_state_width = self.env.observation_space._shape
        self.screen_buffer = [
            np.empty((self.original_state_height, self.original_state_width), dtype=np.uint8),
            np.empty((self.original_state_height, self.original_state_width), dtype=np.uint8),
        ]

    @property
    def observation(self) -> np.ndarray:
        return np.copy(self.state_[:, :, -1])

    @property
    def state(self) -> np.ndarray:
        return jnp.array(self.state_, dtype=jnp.float32)

    def reset(self) -> None:
        obs_, info_ = self.env.reset()

        self.n_steps = 0
        self.n_lives = info_["lives"]  # to terminate on loss life

        self.screen_buffer[0] = obs_
        self.screen_buffer[1].fill(0)

        self.state_ = np.zeros((self.state_height, self.state_width, self.n_stacked_frames), dtype=np.uint8)
        self.state_[:, :, -1] = self.resize()

    def reset_with_noop(self, key):
        self.reset()
        n_noops = jax.random.randint(key, (), 0, 30)  # max_noops = 30
        for _ in range(n_noops):
            _, terminal, _ = self.step(0)
            if terminal:
                self.reset()
        self.n_steps = 0

    def step(self, action: jnp.int8) -> Tuple[float, bool]:
        reward = 0

        for idx_frame in range(self.n_skipped_frames):
            obs_, reward_, game_over, _, info_ = self.env.step(action)

            # we terminate in RB on loss of life but end episode on game_over
            terminal = game_over or (info_["lives"] < self.n_lives)
            self.n_lives = info_["lives"]

            reward += reward_

            if terminal:
                self.state_.fill(0)
                break

            if idx_frame >= self.n_skipped_frames - 2:
                self.screen_buffer[idx_frame - (self.n_skipped_frames - 2)] = obs_

        self.state_ = np.roll(self.state_, -1, axis=-1)
        self.state_[:, :, -1] = self.pool_and_resize()

        self.n_steps += 1

        return reward, terminal, game_over

    def pool_and_resize(self) -> np.ndarray:
        np.maximum(self.screen_buffer[0], self.screen_buffer[1], out=self.screen_buffer[0])

        return self.resize()

    def resize(self):
        return np.asarray(
            cv2.resize(self.screen_buffer[0], (self.state_width, self.state_height), interpolation=cv2.INTER_AREA),
            dtype=np.uint8,
        )
