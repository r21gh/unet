import jax.numpy as jnp
from jax import random
from jaxtest import JaxTestCase
from jax import jit

from ..src.model import Encoder, Decoder

class ModelTest(JaxTestCase):

    def test_encoder_output_shape(self):
        encoder = Encoder(features=64, training=False)
        x = random.normal(key=self.rng, shape=(1, 256, 256, 3))
        params = encoder.init(self.rng, x)

        z_skips = encoder.apply(params, x)
        self.assertEqual(z_skips[0].shape, (1, 256, 256, 64))
        self.assertEqual(z_skips[1].shape, (1, 128, 128, 128))
        self.assertEqual(z_skips[2].shape, (1, 64, 64, 256))
        self.assertEqual(z_skips[3].shape, (1, 32, 32, 512))
        self.assertEqual(z_skips[4].shape, (1, 16, 16, 1024))

    def test_decoder_output_shape(self):
        decoder = Decoder(features=64, training=False)
        z1 = random.normal(key=self.rng, shape=(1, 256, 256, 64))
        z2 = random.normal(key=self.rng, shape=(1, 128, 128, 128))
        z3 = random.normal(key=self.rng, shape=(1, 64, 64, 256))
        z4 = random.normal(key=self.rng, shape=(1, 32, 32, 512))
        z5 = random.normal(key=self.rng, shape=(1, 16, 16, 1024))
        params = decoder.init(self.rng, [z1, z2, z3, z4, z5])

        y = decoder.apply(params, [z1, z2, z3, z4, z5])
        self.assertEqual(y.shape, (1, 256, 256, 3))
