import argparse
from functools import wraps
from typing import Callable, List


def output_added_arguments(add_algo_arguments: Callable) -> Callable:
    @wraps(add_algo_arguments)
    def decorated(parser: argparse.ArgumentParser) -> List[str]:
        unfiltered_old_arguments = list(parser._option_string_actions.keys())

        add_algo_arguments(parser)

        unfiltered_arguments = list(parser._option_string_actions.keys())
        unfiltered_added_arguments = [
            argument for argument in unfiltered_arguments if argument not in unfiltered_old_arguments
        ]

        return [
            argument.strip("-")
            for argument in unfiltered_added_arguments
            if argument.startswith("--") and argument not in ["--help"]
        ]

    return decorated


@output_added_arguments
def add_base_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-en",
        "--experiment_name",
        help="Experiment name.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-s",
        "--seed",
        help="Seed of the experiment.",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-dw",
        "--disable_wandb",
        help="Disable wandb.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-f",
        "--features",
        nargs="*",
        help="List of features for the Q-networks.",
        type=int,
        default=[64, 128, 128, 2048],
    )
    parser.add_argument(
        "-rbc",
        "--replay_buffer_capacity",
        help="Replay Buffer capacity.",
        type=int,
        default=200_000,
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        help="Batch size for training.",
        type=int,
        default=32,
    )
    parser.add_argument(
        "-minn",
        "--min_update_horizon",
        help="Minimum value of n in n-step TD update.",
        type=int,
        default=3,
    )
    parser.add_argument(
        "-maxn",
        "--max_update_horizon",
        help="Maximum value of n in n-step TD update.",
        type=int,
        default=10,
    )
    parser.add_argument(
        "-min_gamma",
        "--min_gamma",
        help="Min discounting factor.",
        type=float,
        default=0.97,
    )
    parser.add_argument(
        "-max_gamma",
        "--max_gamma",
        help="Max discounting factor.",
        type=float,
        default=0.997,
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        help="Learning rate.",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "-horizon",
        "--horizon",
        help="Horizon for truncation.",
        type=int,
        default=27_000,
    )
    parser.add_argument(
        "-nss",
        "--n_sampling_steps",
        help="Number of sampling steps.",
        type=int,
        default=100_000,
    )
    parser.add_argument(
        "-utd",
        "--update_to_data",
        help="Number of gradient steps per sampling steps.",
        type=int,
        default=2,
    )
    parser.add_argument(
        "-tau",
        "--tau",
        help="Soft target update parameter.",
        type=float,
        default=0.005,
    )
    parser.add_argument(
        "-nis",
        "--n_initial_samples",
        help="Number of initial samples before the training starts.",
        type=int,
        default=2_000,
    )
    parser.add_argument(
        "-ee",
        "--epsilon_end",
        help="Ending value for the linear decaying epsilon used for exploration.",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "-ed",
        "--epsilon_duration",
        help="Duration of epsilon's linear decay used for exploration.",
        type=float,
        default=2_001,
    )
    parser.add_argument(
        "-nb",
        "--n_bins",
        help="Number of bins composing the histogram.",
        type=int,
        default=51,
    )
    parser.add_argument(
        "-ghds",
        "--gamma_horizon_decay_steps",
        help="Number of gradient steps to vary n and gamma over.",
        type=float,
        default=10_000,
    )
    parser.add_argument(
        "-rf",
        "--reset_frequency",
        help="Number of sampling steps before resetting the network.",
        type=int,
        default=20_000,
    )
    parser.add_argument(
        "-eval",
        "--eval",
        help="Run evaluation.",
        default=False,
        action="store_true",
    )


@output_added_arguments
def add_bbf_arguments(parser: argparse.ArgumentParser):
    pass
