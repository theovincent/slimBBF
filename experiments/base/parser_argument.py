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
        default=[16, 32, 32, 512],
    )
    parser.add_argument(
        "-rbc",
        "--replay_buffer_capacity",
        help="Replay Buffer capacity.",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        help="Batch size for training.",
        type=int,
        default=32,
    )
    parser.add_argument(
        "-n",
        "--update_horizon",
        help="Value of n in n-step TD update.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-gamma",
        "--gamma",
        help="Discounting factor.",
        type=float,
        default=0.99,
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        help="Learning rate.",
        type=float,
        default=3e-4,
    )
    parser.add_argument(
        "-horizon",
        "--horizon",
        help="Horizon for truncation.",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "-ne",
        "--n_epochs",
        help="Number of epochs to perform.",
        type=int,
        default=50,
    )
    parser.add_argument(
        "-ntspe",
        "--n_training_steps_per_epoch",
        help="Number of training steps per epoch.",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "-utd",
        "--update_to_data",
        help="Number of data points to collect per online Q-network update.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-tuf",
        "--target_update_frequency",
        help="Number of training steps before updating the target Q-network.",
        type=int,
        default=200,
    )
    parser.add_argument(
        "-nis",
        "--n_initial_samples",
        help="Number of initial samples before the training starts.",
        type=int,
        default=1_000,
    )
    parser.add_argument(
        "-ee",
        "--epsilon_end",
        help="Ending value for the linear decaying epsilon used for exploration.",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "-ed",
        "--epsilon_duration",
        help="Duration of epsilon's linear decay used for exploration.",
        type=float,
        default=1_000,
    )


def add_categorical_loss_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-nb",
        "--n_bins",
        help="Number of bins composing the histogram.",
        type=int,
        default=51,
    )
    parser.add_argument(
        "-minn",
        "--min_value",
        help="Value of the lowest learnable value of the target.",
        type=float,
        default=-100,
    )
    parser.add_argument(
        "-maxn",
        "--max_value",
        help="Value of the highest learnable value of the target.",
        type=float,
        default=100,
    )


@output_added_arguments
def add_dqn_arguments(parser: argparse.ArgumentParser):
    pass


@output_added_arguments
def add_per_arguments(parser: argparse.ArgumentParser):
    pass


@output_added_arguments
def add_rainbow_arguments(parser: argparse.ArgumentParser):
    add_categorical_loss_arguments(parser)


@output_added_arguments
def add_bbf_arguments(parser: argparse.ArgumentParser):
    add_categorical_loss_arguments(parser)
    parser.add_argument(
        "-min_gamma",
        "--min_gamma",
        help="Minimum discounting factor to start exp growth from.",
        type=float,
        default=0.97,
    )
    parser.add_argument(
        "-max_n",
        "--max_update_horizon",
        help="Maximum value of n in n-step TD update to start decay from.",
        type=int,
        default=10,
    )
    parser.add_argument(
        "-nupts",
        "--n_updates_per_train_step",
        help="Number of online Q-network updates at every train step.",
        type=int,
        default=2,
    )
    parser.add_argument(
        "-hcs",
        "--horizon_cycle_steps",
        help="Number of steps to vary n and gamma over.",
        type=float,
        default=5_000,
    )
    parser.add_argument(
        "-tut",
        "--target_update_tau",
        help="Fraction for Polyak averaging at target update.",
        type=float,
        default=0.005,
    )
    parser.add_argument(
        "-rf",
        "--reset_frequency",
        help="Number of steps before resetting the network (shrink and perturb).",
        type=int,
        default=20_000,
    )
    parser.add_argument(
        "-nras",
        "--no_resets_after_step",
        help="Maximum step to perform resets upto.",
        type=int,
        default=100_000,
    )
    parser.add_argument(
        "-sf",
        "--shrink_factor",
        help="Shrink scale at resetting.",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "-pf",
        "--perturb_factor",
        help="Perturb scale at resetting.",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "-sprj",
        "--spr_jumps",
        help="SPR loss window size.",
        type=int,
        default=5,
    )
    parser.add_argument(
        "-sprw",
        "--spr_weight",
        help="Scaling factor for SPR loss.",
        type=float,
        default=5,
    )
    parser.add_argument(
        "-tfas",
        "--target_for_action_selection",
        help="Use target network to collect samples.",
        default=False,
        action="store_true",
    )
