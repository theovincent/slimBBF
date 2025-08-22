import ale_py
import gymnasium as gym
import numpy as np
import jax
import jax.numpy as jnp
import cv2
from typing import Tuple


class AtariEval:
    def __init__(self, name: str, sticky_actions: bool, n_envs: int, key: jax.Array) -> None:
        self.name = name
        self.n_envs = n_envs
        self.state_height, self.state_width = (84, 84)
        self.n_stacked_frames = 4
        self.n_skipped_frames = 4

        gym.register_envs(ale_py)

        # Vectorized environments
        self.envs = gym.vector.AsyncVectorEnv(
            [
                lambda: gym.make(
                    f"ALE/{self.name}-v5",
                    full_action_space=False,
                    frameskip=1,
                    repeat_action_probability=0.25 if sticky_actions else 0.0,
                    max_num_frames_per_episode=108_000,
                )
                for _ in range(n_envs)
            ]
        )

        self.n_actions = self.envs.single_action_space.n
        self.original_state_height, self.original_state_width, _ = self.envs.single_observation_space.shape

        self.screen_buffers = np.zeros(
            (n_envs, 2, self.original_state_height, self.original_state_width), dtype=np.uint8
        )
        self.states = np.zeros((n_envs, self.state_height, self.state_width, self.n_stacked_frames), dtype=np.uint8)
        self.n_steps = np.zeros(n_envs, dtype=np.int32)
        self.n_lives = np.zeros(n_envs, dtype=np.int32)
        self.termination_mask = np.zeros(n_envs, dtype=np.uint8)

        noop_key = jax.random.split(key, n_envs)
        for i in range(n_envs):
            self.reset_with_noop_warmup(noop_key[i], i)

    def reset(self, env_id) -> None:
        self.envs[env_id].reset()
        self.n_lives[env_id] = self.envs[env_id].ale.lives()  # to terminate on loss life

        self.envs[env_id].ale.getScreenGrayscale(self.screen_buffer[env_id, 0])
        self.screen_buffer[env_id, 1].fill(0)
        self.states[env_id, :, :, -1] = self.resize(self.screen_buffers[env_id, 0])

    def reset_with_noop_warmup(self, key, env_id):
        self.reset(env_id)
        n_noops = jax.random.randint(key, (), 0, 30)  # max_noops for warmup = 30
        for _ in range(n_noops):
            _, terminal = self.step(0)
            if terminal:
                self.reset()

    @property
    def states(self) -> jnp.ndarray:
        return jnp.array(self.states, dtype=jnp.float32)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rewards = np.zeros(self.n_envs, dtype=np.float32)

        for idx_frame in range(self.n_skipped_frames):
            _, rewards_, terminals_, truncations_, _ = self.envs.step(actions)
            rewards += rewards_ * (1 - self.termination_mask)
            self.termination_mask = self.termination_mask | terminals_ | truncations_

            if idx_frame >= self.n_skipped_frames - 2:
                t = idx_frame - (self.n_skipped_frames - 2)
                for i in range(self.n_envs):
                    self.envs.envs[i].unwrapped.ale.getScreenGrayscale(self.screen_buffers[i, t])

        pooled = np.maximum(self.screen_buffers, axis=1)
        resized = np.stack([self.resize(pooled[i]) for i in range(self.n_envs)], axis=0)

        self.states = np.roll(self.states, -1, axis=-1)
        self.states[:, :, :, -1] = resized
        self.n_steps += 1

        return rewards

    def resize(self, frame: np.ndarray) -> np.ndarray:
        return np.asarray(
            cv2.resize(frame, (self.state_width, self.state_height), interpolation=cv2.INTER_AREA),
            dtype=np.uint8,
        )
