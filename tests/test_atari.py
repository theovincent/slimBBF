import os
import shutil
import subprocess
import unittest


class TestAtari(unittest.TestCase):
    def test_bbf(self):
        save_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../experiments/atari/exp_output/_test_bbf_Pong"
        )
        if os.path.exists(save_path):
            shutil.rmtree(save_path)

        returncode = subprocess.run(
            [
                "python3",
                "experiments/atari/bbf.py",
                "--experiment_name",
                "_test_bbf_Pong",
                "--seed",
                "1",
                "--disable_wandb",
                "--features",
                "2",
                "3",
                "1",
                "15",
                "--replay_buffer_capacity",
                "100",
                "--batch_size",
                "3",
                "--learning_rate",
                "1e-4",
                "--horizon",
                "10",
                "--n_sampling_steps",
                "2",
                "--n_initial_samples",
                "3",
                "--epsilon_end",
                "0.01",
                "--epsilon_duration",
                "4",
            ]
        ).returncode
        assert returncode == 0, "The command should not have raised an error."

        shutil.rmtree(save_path)
