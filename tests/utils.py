from typing import Tuple
from functools import partial
import jax
import jax.numpy as jnp

from slimbbf.sample_collection.replay_buffer import ReplayElement
from slimbbf.sample_collection.subseq_replay_buffer import SubsequenceReplayElement


class Generator:
    def __init__(self, batch_size: int, observation_dim: Tuple[int], n_actions: int) -> None:
        self.batch_size = batch_size
        self.observation_dim = observation_dim
        self.n_actions = n_actions

    @partial(jax.jit, static_argnames="self")
    def sample(
        self,
        key: jax.random.PRNGKey,
    ) -> ReplayElement:
        state = jax.random.uniform(key, self.observation_dim)
        action = jax.random.randint(key, (), minval=0, maxval=self.n_actions, dtype=jnp.int8)
        _, key_ = jax.random.split(key)
        reward = jax.random.uniform(key_)
        terminal = jax.random.randint(key_, (), 0, 2)
        next_state = jax.random.uniform(key_, self.observation_dim)
        return ReplayElement(
            state,  # state
            action,  # action
            reward,  # reward
            next_state,  # next_state
            terminal,  # terminal
        )

    @partial(jax.jit, static_argnames="self")
    def samples(self, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray]:
        return jax.vmap(self.sample)(jax.random.split(key, self.batch_size))

    @partial(jax.jit, static_argnames="self")
    def state(self, key: jax.random.PRNGKey) -> jnp.ndarray:
        return jax.random.uniform(key, self.observation_dim)

    @partial(jax.jit, static_argnames="self")
    def states(self, key: jax.random.PRNGKey) -> jnp.ndarray:
        return jax.random.uniform(key, (self.batch_size,) + self.observation_dim)

    def sample_subseq_replay_buffer(
        self,
        key: jax.random.PRNGKey,
    ) -> ReplayElement:
        spr_window = jax.random.randint(key, (), minval=1, maxval=10)
        states_stack = jax.random.uniform(key, (spr_window + 1, *self.observation_dim))
        actions_stack = jax.random.randint(key, (spr_window + 1,), minval=0, maxval=self.n_actions, dtype=jnp.int8)
        _, key_ = jax.random.split(key)
        reward = jax.random.uniform(key_)
        terminal = jax.random.randint(key_, (), 0, 2)
        next_state = jax.random.uniform(key_, self.observation_dim)
        same_trajectory_mask = jnp.ones((spr_window + 1,))
        if terminal:
            same_trajectory_mask = same_trajectory_mask.at[1].set(0)
        same_trajectory_mask = same_trajectory_mask.at[
            jax.random.randint(key, (), minval=1, maxval=spr_window).item()
        ].set(0)
        same_trajectory_mask = same_trajectory_mask.cumprod()
        return SubsequenceReplayElement(
            states_stack,  # state
            actions_stack,  # action
            reward,  # reward
            next_state,  # next_state
            terminal,  # terminal
            same_trajectory_mask,
        )
