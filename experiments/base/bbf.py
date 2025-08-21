import jax
import optax
from tqdm import trange

from experiments.base.utils import save_data
from slimbbf.algorithms.bbf import BBF
from slimbbf.sample_collection.subseq_replay_buffer import SubsequenceReplayBuffer
from slimbbf.sample_collection.utils import collect_single_sample


def train(key: jax.Array, p: dict, agent: BBF, env, rb: SubsequenceReplayBuffer):
    epsilon_schedule = optax.linear_schedule(1.0, p["epsilon_end"], p["epsilon_duration"], p["n_initial_samples"])
    env.reset()
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
        EVALUATE THE MODEL WITH 100 EPISODES!!!
