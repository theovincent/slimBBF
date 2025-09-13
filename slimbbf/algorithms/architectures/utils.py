import jax
import jax.numpy as jnp
import numpy as np


def random_crop(key, img, cropped_shape):
    key_width, key_height = jax.random.split(key)
    width = jax.random.randint(key_width, (), 0, img.shape[0] - cropped_shape[0])
    height = jax.random.randint(key_height, (), 0, img.shape[1] - cropped_shape[1])
    # Only crop along width and height axes
    return jax.lax.dynamic_slice(img, [width, height, 0], cropped_shape)


@jax.jit
def normalize_and_augment(state, key):
    # Padding, cropping, and intensity augmentations
    x = jnp.array(state, ndmin=4) / 255.0

    # Only pad along width and height axes
    x_shape = x.shape
    x = x.reshape(-1, *x_shape[-3:])
    x_padded = jnp.pad(x, [(0, 0), (4, 4), (4, 4), (0, 0)], "edge")
    crop_key, intensity_key = jax.random.split(key)
    x_cropped = jax.vmap(random_crop, in_axes=(0, 0, None))(
        jax.random.split(crop_key, x.shape[0]), x_padded, x.shape[-3:]
    )
    x_augmented = x_cropped * (
        1.0 + (0.05 * jnp.clip(jax.random.normal(intensity_key, shape=(x.shape[0], 1, 1, 1)), -2.0, 2.0))
    )
    return x_augmented.reshape(x_shape).squeeze()


def exponential_scheduler(decay_period, initial_value, final_value):
    return lambda step: (
        initial_value * np.power(final_value / initial_value, (step / decay_period))
        if step <= decay_period
        else final_value
    )


def reverse_exponential_scheduler(decay_period, initial_value, final_value):
    schedule = exponential_scheduler(decay_period, 1 - initial_value, 1 - final_value)
    return lambda step: 1 - schedule(step)
