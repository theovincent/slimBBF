import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from slimbbf.sample_collection.subsequence_replay_buffer import PrioritizedJaxSubsequenceParallelEnvReplayBuffer


@partial(jax.jit, static_argnames=("best_action_fn", "n_actions", "epsilon_fn"))
def select_action(best_action_fn, params, state, key, n_actions, epsilon_fn, n_sampling_steps):
    uniform_key, action_key = jax.random.split(key)
    return jnp.where(
        jax.random.uniform(uniform_key) <= epsilon_fn(n_sampling_steps),  # if uniform < epsilon,
        jax.random.randint(action_key, (), 0, n_actions),  # take random action
        best_action_fn(params, state),  # otherwise, take a greedy action
    )


@partial(jax.jit, static_argnames=("best_action_fn", "n_actions", "epsilon_fn"))
def select_action_eval(best_action_fn, params, states, key, n_actions, epsilon_fn):
    selection_keys = jax.random.split(key, states.shape[0])
    return jax.vmap(select_action, in_axes=(None, None, 0, 0, None, None, None))(
        best_action_fn, params, states, selection_keys, n_actions, epsilon_fn, 0
    )


def collect_single_sample(
    key, env, agent, rb: PrioritizedJaxSubsequenceParallelEnvReplayBuffer, p, epsilon_schedule, n_sampling_steps: int
):
    action = select_action(
        agent.best_action, agent.target_params, env.state, key, env.n_actions, epsilon_schedule, n_sampling_steps
    ).item()

    obs = env.observation
    reward, absorbing, game_over = env.step(action)

    is_truncation = env.n_steps >= p["horizon"]
    rb.add(
        observation=obs[None, :],
        action=np.array([action]),
        reward=np.array([rb.clipping(reward)]),
        terminal=np.array([absorbing]),
        priority=np.array([rb.sum_tree.max_recorded_priority]),
        episode_end=np.array([is_truncation]),
    )  # loss of live also absorbing for RB

    # On absorbing (life loss), the episode continues. We reset and log only when game over or truncation
    if game_over or is_truncation:
        _, noop_key = jax.random.split(key)
        env.reset_with_noop(noop_key)

    return reward, game_over or is_truncation
