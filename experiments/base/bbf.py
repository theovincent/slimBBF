import jax
import numpy as np
import optax

from experiments.base.utils import save_data
from slimbbf.algorithms.bbf import BBF
from slimbbf.sample_collection.subseq_replay_buffer import SubsequenceReplayBuffer
from slimbbf.sample_collection.utils import collect_single_sample, select_action_eval


def train(key: jax.Array, p: dict, agent: BBF, env, env_eval, rb: SubsequenceReplayBuffer):
    epsilon_schedule = optax.linear_schedule(1.0, p["epsilon_end"], p["epsilon_duration"], p["n_initial_samples"])
    key, noop_key = jax.random.split(key)
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
        if has_reset:
            print(
                f"\n{n_sampling_steps} sampling steps: Return {episode_returns[-1]} after {episode_lengths[-1]} steps.\n",
                flush=True,
            )
            p["wandb"].log(
                {
                    "n_sampling_steps": n_sampling_steps,
                    "performances/train_episode_return": episode_returns[-1],
                    "performances/train_episode_length": episode_lengths[-1],
                    **agent.get_logs(),
                }
            )
            episode_returns.append(0)
            episode_lengths.append(0)

        if n_sampling_steps >= p["n_initial_samples"]:
            for _ in range(p["update_to_data"]):
                agent.update_online_params(rb)

            # evaluate every 20K steps
            if n_sampling_steps % 20_000 == 0:
                key, eval_key = jax.random.split(key)
                eval_episode_returns, eval_episode_lengths = evaluate(
                    eval_key, p, agent, env_eval(n_envs=10 if n_sampling_steps < p["n_sampling_steps"] else 100)
                )
                p["wandb"].log(
                    {
                        "n_sampling_steps": n_sampling_steps,
                        "performances/eval_avg_return": np.mean(eval_episode_returns),
                        "performances/eval_avg_length": np.mean(eval_episode_lengths),
                    }
                )
                eval_returns.append(eval_episode_returns)
                eval_lengths.append(eval_episode_lengths)

            # avoid resetting on last 2 iterations
            if (
                n_sampling_steps % p["reset_frequency"] == 0
                and n_sampling_steps < p["n_sampling_steps"] - p["reset_frequency"]
            ):
                agent.reset_params()

    save_data(p, episode_returns, episode_lengths, agent.get_model())
    save_data(p, eval_returns, eval_lengths, None)


def evaluate(key: jax.Array, p: dict, agent: BBF, env):
    key, reset_key = jax.random.split(key)
    env.reset_with_noop(reset_key)
    episode_termination = env.game_over_mask  # needed for considering rewards,length until env.game_over_mask
    episode_returns = np.zeros(env.n_envs)
    episode_lengths = np.zeros(env.n_envs)
    epsilon_fn = lambda _: 0.001

    while not episode_termination.all() and env.n_steps < p["horizon"]:
        key, actions_key = jax.random.split(key)

        actions = select_action_eval(
            agent.best_action, agent.target_params, env.states, actions_key, env.n_actions, epsilon_fn
        )
        rewards = env.step(np.array(actions))

        # episode.termination changes here, so we use episode_termination
        episode_returns += rewards * (1 - episode_termination)
        episode_lengths += 1 - episode_termination
        episode_termination = env.game_over_mask

    return episode_returns.tolist(), episode_lengths.tolist()
