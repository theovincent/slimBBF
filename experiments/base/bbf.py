import os
import json
import jax
import numpy as np
import optax
from tqdm import trange

from experiments.base.utils import save_data
from slimbbf.algorithms.bbf import BBF
from slimbbf.sample_collection.subseq_replay_buffer import SubsequenceReplayBuffer
from slimbbf.sample_collection.utils import collect_single_sample, select_action


def train(key: jax.Array, p: dict, agent: BBF, env, rb: SubsequenceReplayBuffer):
    epsilon_schedule = optax.linear_schedule(1.0, p["epsilon_end"], p["epsilon_duration"], p["n_initial_samples"])
    noop_key, key = jax.random.split(key)
    env.reset_with_noop(noop_key)
    episode_returns = [0]
    episode_lengths = [0]

    for n_sampling_steps in trange(1, p["n_sampling_steps"] + 1, miniters=10000, maxinterval=1000):
        key, explore_key = jax.random.split(key)
        reward, has_reset = collect_single_sample(explore_key, env, agent, rb, p, epsilon_schedule, n_sampling_steps)

        episode_returns[-1] += reward
        episode_lengths[-1] += 1
        if has_reset:
            print(
                f"\{n_sampling_steps} sampling steps: Return {episode_returns[-1]} after {episode_lengths[-1]} steps.\n",
                flush=True,
            )
            p["wandb"].log(
                {
                    "n_sampling_steps": n_sampling_steps,
                    "performances/episode_return": episode_returns[-1],
                    "performances/episode_length": episode_lengths[-1],
                    **agent.get_logs(),
                }
            )
            episode_returns.append(0)
            episode_lengths.append(0)

        if n_sampling_steps >= p["n_initial_samples"]:
            for _ in range(p["update_to_data"]):
                agent.update_online_params(rb)
            # avoid resetting on last iteration
            agent.reset_params(n_sampling_steps if n_sampling_steps < p["n_sampling_steps"] else 1)

        save_data(p, episode_returns, episode_lengths, agent.get_model())


def eval(key: jax.Array, p: dict, agent: BBF, env):
    episode_termination = np.zeros((env.n_envs,), dtype=np.uint8)
    episode_returns = np.zeros((env.n_envs,), dtype=np.float32)
    episode_lengths = np.zeros((env.n_envs,), dtype=np.uint32)
    while not env.termination_mask.all():
        actions_key, key = jax.random.split(key)
        actions_key = jax.random.split(actions_key, env.n_envs)
        actions = jax.vmap(select_action, in_axes=(None, None, 0, 0, None, None, None))(
            agent.best_action, agent.params, env.states, actions_key, env.n_actions, lambda _: 0.001, 1
        )
        rewards = env.step(actions)
        episode_returns += rewards * (1 - episode_termination)
        episode_lengths += np.ones((env.n_envs,), dtype=np.uint32) * (1 - episode_termination)
        episode_termination |= env.termination_mask

    os.makedirs(os.path.join(p["save_path"], "eval_episode_returns_and_lengths"), exist_ok=True)
    episode_returns_and_lengths_path = os.path.join(
        p["save_path"], f"eval_episode_returns_and_lengths/{p['seed']}.json"
    )
    json.dump(
        {"episode_lengths": episode_lengths, "episode_returns": episode_returns},
        open(episode_returns_and_lengths_path, "w"),
        indent=4,
    )
