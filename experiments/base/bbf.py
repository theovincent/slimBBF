import jax
import numpy as np
import optax

from experiments.base.utils import save_data
from slimbbf.algorithms.bbf import BBF
from slimbbf.sample_collection.subseq_replay_buffer import SubsequenceReplayBuffer
from slimbbf.sample_collection.utils import collect_single_sample, select_action


def train(key: jax.Array, p: dict, agent: BBF, env, env_eval, rb: SubsequenceReplayBuffer):
    epsilon_schedule = optax.linear_schedule(1.0, p["epsilon_end"], p["epsilon_duration"], p["n_initial_samples"])
    noop_key, key = jax.random.split(key)
    env.reset_with_noop(noop_key)
    episode_returns = [0]
    episode_lengths = [0]
    eval_returns = []
    eval_lengths = []

    for n_sampling_steps in range(1, p["n_sampling_steps"] + 1):
        key, explore_key = jax.random.split(key)
        reward, has_reset = collect_single_sample(explore_key, env, agent, rb, p, epsilon_schedule, n_sampling_steps)

        episode_returns[-1] += reward
        episode_lengths[-1] += 1
        if has_reset or n_sampling_steps == p["n_sampling_steps"]:
            print(
                f"\n{n_sampling_steps} sampling steps: Return {episode_returns[-1]} after {episode_lengths[-1]} steps.\n",
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

            # evaluate every 20K steps (includes evaluation at the end)
            if n_sampling_steps % 20_000 == 0:
                eval_key, key = jax.random.split(key)
                eval_episode_returns, eval_episode_lengths = eval(eval_key, p, agent, env_eval())
                p["wandb"].log(
                    {
                        "n_sampling_steps": n_sampling_steps,
                        "performances/eval_avg_return": np.mean(eval_episode_returns),
                        "performances/eval_avg_length": np.mean(episode_lengths[-1]),
                    }
                )
                eval_returns.append(eval_episode_returns)
                eval_lengths.append(eval_episode_lengths)

            # avoid resetting on last iteration
            agent.reset_params(n_sampling_steps if n_sampling_steps < p["n_sampling_steps"] else 1)

        save_data(p, episode_returns, episode_lengths, agent.get_model())

    save_data(p, eval_returns, eval_lengths, None)


def eval(key: jax.Array, p: dict, agent: BBF, env):
    episode_termination = env.termination_mask  # needed for considering rewards,length until env.termination_mask
    episode_returns = np.zeros((env.n_envs,), dtype=np.float32)
    episode_lengths = np.zeros((env.n_envs,), dtype=np.uint32)
    while not episode_termination.all() and env.n_steps < p["horizon"]:
        actions_key, key = jax.random.split(key)
        actions_key = jax.random.split(actions_key, env.n_envs)
        actions = jax.vmap(select_action, in_axes=(None, None, 0, 0, None, None, None))(
            agent.best_action, agent.params, env.states, actions_key, env.n_actions, lambda _: 0.001, 1
        )
        rewards = env.step(np.array(actions))  # episode.termination changes here, so we use episode_termination
        episode_returns += rewards * (1 - episode_termination)
        episode_lengths += (np.ones((env.n_envs,)) * (1 - episode_termination)).astype(np.uint32)
        episode_termination = env.termination_mask

    return episode_returns.tolist(), episode_lengths.tolist()
