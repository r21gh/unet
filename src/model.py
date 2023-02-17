import jax
from jax import numpy as jnp
from flax import linen as fnn


class Encoder(fnn.Module):
    features: int = 64
    training: bool = True

    @fnn.compact
    def __call__(self, x):
        z = x
        skips = []
        for i in range(4):
            z = fnn.Conv(self.features * 2 ** i, kernel_size=(3, 3))(z)
            z = fnn.relu(z)
            z = fnn.Conv(self.features * 2 ** i, kernel_size=(3, 3))(z)
            z = fnn.BatchNorm(use_running_average=not self.training)(z)
            z = fnn.relu(z)
            if i > 2:
                z = fnn.Dropout(0.5, deterministic=False)(z)
            if i < 3:
                z = fnn.max_pool(z, window_shape=(2, 2), strides=(2, 2))
            skips.append(z)

        return skips


class Decoder(fnn.Module):
    features: int = 64
    training: bool = True

    @fnn.compact
    def __call__(self, skips):
        z = skips[-1]
        for i in range(3):
            z = jax.image.resize(z, shape=(z.shape[0], z.shape[1] * 2, z.shape[2] * 2, z.shape[3]), method='nearest')
            z = jnp.concatenate([skips[-i-2], z], axis=3)
            z = fnn.Conv(self.features * 2 ** (3 - i), kernel_size=(3, 3))(z)
            z = fnn.relu(z)
            z = fnn.Conv(self.features * 2 ** (3 - i), kernel_size=(3, 3))(z)
            z = fnn.BatchNorm(use_running_average=not self.training)(z)
            z = fnn.relu(z)

        z = fnn.Conv(1, kernel_size=(1, 1))(z)
        z = fnn.sigmoid(z)
        return z


class UNet(fnn.Module):
    features: int = 64
    training: bool = True

    @fnn.compact
    def __call__(self, x):
        z_skips = Encoder(self.training)(x)
        y = Decoder(self.training)(z_skips)

        return y


