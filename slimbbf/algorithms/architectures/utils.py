import jax
import jax.numpy as jnp


def random_crop(key, img, cropped_shape):
    key_weight, key_heigth = jax.random.split(key, 2)
    weight = jax.random.randint(key_weight, (), 0, img.shape[0] - cropped_shape.shape[0])
    heigth = jax.random.randint(key_heigth, (), 0, img.shape[1] - cropped_shape.shape[1])
    # Only crop along weight and height axes
    return jax.lax.dynamic_slice(img, [weight, heigth, 0], cropped_shape)


@jax.jit
def normalize_and_augment(state, key):
    # Padding, cropping, and intensity augmentations
    x = jnp.array(state, ndmin=4) / 255.0

    # Only pad along weight and heigth axes
    x_padded = jnp.pad(x, [(0, 0), (4, 4), (4, 4), (0, 0)], "edge")
    crop_key, intensity_key = jax.random.split(key)
    x_cropped = jax.vmap(random_crop)(jax.random.split(crop_key, x.shape[0]), x_padded, x.shape)
    x_augmented = x_cropped * (
        1.0 + (0.05 * jnp.clip(jax.random.normal(intensity_key, shape=(x.shape[0], 1, 1, 1)), -2.0, 2.0))
    )
    return x_augmented.squeeze(axis=0)


def exponential_scheduler(decay_period, initial_value, final_value):
    return lambda step: (
        initial_value * jnp.power(final_value / initial_value, (step / decay_period))
        if step <= decay_period
        else final_value
    )


def reverse_exponential_scheduler(decay_period, initial_value, final_value):
    schedule = exponential_scheduler(decay_period, 1 - initial_value, 1 - final_value)
    return lambda step: 1 - schedule(step)
