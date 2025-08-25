from functools import partial
import unittest
import numpy as np
import jax
import jax.numpy as jnp
import optax

from slimbbf.algorithms.bbf import BBF
from tests.utils import Generator


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
            max_gamma=0.99,
            min_update_horizon=3,
            max_update_horizon=10,
            gamma_horizon_decay_steps=10,
            tau=0.005,
            reset_frequency=1,
            spr_steps=5,
        )

        self.generator = Generator(None, self.observation_dim, self.n_actions)

    def test_compute_target(self) -> None:
        print(f"-------------- Random key {self.random_seed} --------------")
        sample = self.generator.sample_subseq_replay_buffer(self.key)
        gamma = jax.random.uniform(self.key)
        update_horizon = jax.random.randint(self.key, (), minval=1, maxval=10)
        computed_target_probs = self.q.compute_target(self.q.params, sample, gamma**update_horizon)

        target_prob_actions = jax.nn.softmax(self.q.network.apply(self.q.params, sample.next_state))
        target_probs = target_prob_actions[jnp.argmax(jax.nn.softmax(target_prob_actions) @ self.q.bins)]
        target_locations_ = sample.reward + (1 - sample.is_terminal) * (gamma**update_horizon) * self.q.bins
        targets_locations = jnp.clip(target_locations_, self.q.bins[0], self.q.bins[-1])

        def projection(bin_location):
            distances_to_bin = jnp.abs(targets_locations - bin_location) / (self.q.bins[1] - self.q.bins[0])
            return jnp.dot((1 - jnp.minimum(distances_to_bin, 1)), target_probs)

        actual_target_probs = jax.vmap(projection)(self.q.bins)

        np.testing.assert_array_equal(actual_target_probs, computed_target_probs)

    def test_loss(self) -> None:
        print(f"-------------- Random key {self.random_seed} --------------")
        sample = self.generator.sample_subseq_replay_buffer(self.key)
        gamma = jax.random.uniform(self.key)
        importance_weight = jax.random.uniform(self.key)
        update_horizon = jax.random.randint(self.key, (), minval=1, maxval=10)

        computed_loss = self.q.loss(self.q.params, self.q.params, sample, importance_weight, gamma**update_horizon)

        target_probs = self.q.compute_target(self.q.params, sample, gamma**update_horizon)
        q_probs, spr_predictions = self.q.network.apply(
            self.q.params, sample.states_stack[0], sample.actions_stack[:-1]
        )
        cross_entropy = importance_weight * optax.softmax_cross_entropy(q_probs, jax.lax.stop_gradient(target_probs))

        spr_targets = jax.vmap(
            partial(self.q.network.apply, method=self.q.network.encode_and_project), in_axes=(None, 0)
        )(self.q.params, sample.states_stack[1:])
        spr_targets = spr_targets / jnp.linalg.norm(spr_targets, axis=-1, keepdims=True)
        spr_predictions = spr_predictions / jnp.linalg.norm(spr_predictions, axis=-1, keepdims=True)
        spr_losses = jnp.square(spr_predictions - jax.lax.stop_gradient(spr_targets)).sum(axis=-1)
        spr_loss = importance_weight * (spr_losses * sample.same_trajectory_mask[1:]).mean()

        self.assertEqual(cross_entropy + 5 * spr_loss, computed_loss[0])
        self.assertEqual(cross_entropy, computed_loss[1])
        self.assertEqual(spr_loss, computed_loss[2])

    def test_best_action(self):
        print(f"-------------- Random key {self.random_seed} --------------")
        state = self.generator.state(self.key)

        computed_best_action = self.q.best_action(self.q.params, state)

        q_probs = jax.nn.softmax(self.q.network.apply(self.q.params, state / 255.0))
        best_action = jnp.argmax(q_probs @ self.q.bins)
        self.assertEqual(best_action, computed_best_action)


if __name__ == "__main__":
    unittest.main()
