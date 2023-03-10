import time
from functools import partial
import jax
import optax

from jax import numpy as jnp
from flax.training.train_state import TrainState
from flax.training import checkpoints

from model import UNet
from data_loader import read_train_data, read_predict_data, save_image
from training import CustomTrainState, train_step
from prediction import predict

CKPT_DIR = 'checkpoints'
IMAGE_SIZE = 512
NUM_EPOCHS = 20

def run():
    train_set = read_train_data()
    unet = UNet()

    init_rngs = {'params': jax.random.PRNGKey(0), 'dropout': jax.random.PRNGKey(1)}

    unet_variables = unet.init(init_rngs, jnp.ones([1, IMAGE_SIZE, IMAGE_SIZE, 3]))

    optimizer = optax.adam(learning_rate=0.001)

    train_state = CustomTrainState.create(apply_fn=unet.apply, params=unet_variables["params"], tx=optimizer,
                                          batch_stats=unet_variables["batch_stats"])

    checkpoints.save_checkpoint(ckpt_dir=CKPT_DIR, target=train_state, step=0, overwrite=True)

    for e in range(NUM_EPOCHS):
        loss_avg = 0
        tic = time.time()
        for x, y in train_set.as_numpy_iterator():
            loss, train_state = train_step(x, y, train_state, True)
            loss_avg += loss

        loss_avg /= len(train_set)
        elapsed = time.time() - tic
        print(f"epoch: {e}, loss: {loss_avg:0.2f}, elapased: {elapsed:0.2f}")

    predict()

if __name__ == '__main__':
    run()