import unittest
import numpy as np
import jax
import jax.numpy as jnp
import optax

from slimbbf.algorithms.bbf import BBF
from tests.utils import Generator

#     key: Any,
#     observation_dim: Any,
#     n_actions: Any,
#     n_bins: int,
#     features: list,
#     learning_rate: float,
#     min_gamma: float,
#     gamma: float,
#     update_horizon: int,
#     max_update_horizon: int,
#     horizon_cycle_steps: int,
#     update_to_data: int,
#     target_update_tau: float,
#     reset_frequency: int,
#     shrink_factor: float,
#     perturb_factor: float,
#     min_value: float,
#     max_value: float,
#     spr_weight: float,
#     adam_eps: float = 1e-8,
#     adam_weight_decay: float = 0.1


class TestBBF(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.random_seed = np.random.randint(1000)
        self.key = jax.random.PRNGKey(self.random_seed)

        key_actions, key_bins, key_feature_1, key_feature_2, key_feature_3, key_feature_4 = jax.random.split(
            self.key, 6
        )
        self.observation_dim = (84, 84, 4)
        self.n_actions = int(jax.random.randint(key_actions, (), minval=2, maxval=10))
        self.n_bins = int(jax.random.randint(key_bins, (), minval=2, maxval=10))
        self.q = BBF(
            key=self.key,
            observation_dim=self.observation_dim,
            n_actions=self.n_actions,
            n_bins=self.n_bins,
            features=[
                jax.random.randint(key_feature_1, (), minval=5, maxval=10),
                jax.random.randint(key_feature_2, (), minval=5, maxval=10),
                jax.random.randint(key_feature_3, (), minval=5, maxval=10),
                jax.random.randint(key_feature_4, (), minval=5, maxval=10),
            ],
            learning_rate=0.001,
            min_gamma=0.94,
            gamma=0.99,
            update_horizon=3,
            max_update_horizon=10,
            horizon_cycle_steps=10,
            update_to_data=2,
            target_update_tau=0.005,
            reset_frequency=1,
            shrink_factor=0.5,
            perturb_factor=0.5,
            min_value=-10,
            max_value=10,
            spr_weight=5,
            adam_eps=0.00015,
            adam_weight_decay=0.1,
        )

        self.generator = Generator(None, self.observation_dim, self.n_actions)

    def test_compute_target(self) -> None:
        print(f"-------------- Random key {self.random_seed} --------------")
        sample = self.generator.sample_subseq_replay_buffer(self.key)
        self.q.current_gamma = jax.random.uniform(self.key)
        self.q.current_update_horizon = jax.random.randint(self.key, (), minval=1, maxval=10)
        computed_target_support, computed_target_prob = self.q.compute_target(self.q.params, sample)

        next_q_logits = self.q.network.apply(self.q.params, sample.next_state)
        target = (
            sample.reward
            + (1 - sample.is_terminal) * (self.q.current_gamma**self.q.current_update_horizon) * self.q.support
        )
        target_prob = jax.nn.softmax(next_q_logits[jnp.argmax(jax.nn.softmax(next_q_logits) @ self.q.support)])
        self.assertEqual(computed_target_support.shape, (self.n_bins,))
        np.testing.assert_array_equal(target, computed_target_support)
        np.testing.assert_array_equal(target_prob, computed_target_prob)

    def test_loss(self) -> None:
        print(f"-------------- Random key {self.random_seed} --------------")
        sample = self.generator.sample_subseq_replay_buffer(self.key)
        self.q.current_gamma = jax.random.uniform(self.key)
        self.q.current_update_horizon = jax.random.randint(self.key, (), minval=1, maxval=10)

        computed_loss = self.q.loss(self.q.params, self.q.params, sample)

        q_logits, spr_predictions = self.q.network.apply(
            self.q.params, sample.states_stack[0], sample.actions_stack[:-1]
        )
        q_logits = q_logits[sample.actions_stack[0]]
        cross_entropy = optax.softmax_cross_entropy(
            q_logits, self.q.project_target_on_support(*self.q.compute_target(self.q.params, sample))
        )
        spr_predictions = spr_predictions / jnp.linalg.norm(spr_predictions, 2, -1, keepdims=True)
        spr_targets = self.q.network.apply(self.q.params, sample.states_stack[1:], method=self.q.network.encode_project)
        spr_targets = spr_targets / jnp.linalg.norm(spr_targets, 2, -1, keepdims=True)

        spr_loss = jnp.power(spr_predictions - jax.lax.stop_gradient(spr_targets), 2).sum(-1)
        spr_loss = (spr_loss * sample.same_trajectory_mask[:-1]).mean(0)
        self.assertEqual(cross_entropy, computed_loss[1])
        self.assertEqual(cross_entropy + self.q.spr_weight * spr_loss, computed_loss[0])

    def test_best_action(self):
        print(f"-------------- Random key {self.random_seed} --------------")
        state = self.generator.state(self.key)

        computed_best_action = self.q.best_action(self.q.params, state)

        q_logits = self.q.network.apply(self.q.params, state)
        best_action = jnp.argmax(jax.nn.softmax(q_logits) @ self.q.support)

        self.assertEqual(q_logits.shape, (self.n_actions, self.n_bins))
        self.assertEqual(best_action, computed_best_action)


if __name__ == "__main__":
    unittest.main()
