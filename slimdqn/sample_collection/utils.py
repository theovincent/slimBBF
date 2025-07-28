import jax
import jax.numpy as jnp
from functools import partial

from slimdqn.sample_collection.replay_buffer import ReplayBuffer


@partial(jax.jit, static_argnames=("best_action_fn", "n_actions", "epsilon_fn"))
def select_action(best_action_fn, params, state, key, n_actions, epsilon_fn, n_training_steps):
    uniform_key, action_key = jax.random.split(key)
    return jnp.where(
        jax.random.uniform(uniform_key) <= epsilon_fn(n_training_steps),  # if uniform < epsilon,
        jax.random.randint(action_key, (), 0, n_actions),  # take random action
        best_action_fn(params, state),  # otherwise, take a greedy action
    )


def collect_single_sample(
    key,
    env,
    agent,
    rb: ReplayBuffer,
    p,
    epsilon_schedule,
    n_training_steps: int,
    target_for_action_selection: bool = False,
):
    action = select_action(
        agent.best_action,
        agent.target_params if target_for_action_selection else agent.params,
        env.state,
        key,
        env.n_actions,
        epsilon_schedule,
        n_training_steps,
    ).item()

    obs = env.observation
    reward, absorbing = env.step(action)

    is_truncation = env.n_steps >= p["horizon"]
    rb.add(
        observation=obs,
        action=action,
        reward=reward if rb._clipping is None else rb._clipping(reward),
        is_terminal=absorbing,
        is_truncation=is_truncation,
    )

    if absorbing or is_truncation:
        env.reset()

    return reward, absorbing or is_truncation
