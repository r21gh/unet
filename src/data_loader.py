import os, glob
import tensorflow as tf
from jax.lib import xla_bridge

PATH = 'datasets/'
BATCH_SIZE = 12
IMAGE_SIZE = 512

def read_train_data():
    x_files = tf.data.Dataset.list_files(PATH + "train/*.png")
    y_files = tf.data.Dataset.list_files(PATH + "mask/*.png")

    def read_image(x_filename, y_filename):
        x_image_string = tf.io.read_file(x_filename)
        y_image_string = tf.io.read_file(y_filename)

        x_image_decoded = tf.image.decode_png(x_image_string, channels=3)
        y_image_decoded = tf.image.decode_png(y_image_string, channels=1)

        x_image_resized = tf.image.resize(x_image_decoded, [IMAGE_SIZE, IMAGE_SIZE])
        y_image_resized = tf.image.resize(y_image_decoded, [IMAGE_SIZE, IMAGE_SIZE])

        x_image_norm = x_image_resized / 255
        y_image_norm = y_image_resized / 255

        return x_image_norm, y_image_norm

    dataset = tf.data.Dataset.zip((x_files, y_files))

    dataset = dataset.map(read_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(1000).batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def read_predict_data():
    def image_generator():
        for filename in glob.iglob(PATH + "test/*.png"):
            with open(filename, "rb") as f:
                image = tf.image.decode_png(f.read(), channels=3)
                image_resized = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
                image_norm = image_resized / 255.0
                yield image_norm

    images_dataset = tf.data.Dataset.from_generator(
        image_generator,
        output_types=tf.float32,
        output_shapes=(IMAGE_SIZE, IMAGE_SIZE, 3),
    )
    images_dataset = images_dataset.batch(BATCH_SIZE)
    images_dataset = images_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return images_dataset


def save_image(image):
    file_path = 'test/result.png'
    image = image * 255

    encode_image = tf.image.encode_png(image, format='rgb', quality=100)

    with open(file_path, 'wb') as fd:
        fd.write(encode_image)