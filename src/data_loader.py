import glob
import os

import tensorflow as tf


class DataLoader:
    
    def __init__(self, path='../datasets/', batch_size=12, image_size=512):
        self.path = path
        self.batch_size = batch_size
        self.image_size = image_size

    
    def read_train_data(self):
        x_files = tf.data.Dataset.list_files(self.path + "train/*.png", shuffle=True)
        y_files = tf.data.Dataset.list_files(self.path + "mask/*.png", shuffle=True)

        def read_image(x_filename, y_filename):
            x_image_string = tf.io.read_file(x_filename)
            y_image_string = tf.io.read_file(y_filename)

            x_image_decoded = tf.image.decode_png(x_image_string, channels=3) # RGB image
            y_image_decoded = tf.image.decode_png(y_image_string, channels=1) # grayscale image

            x_image_resized = tf.image.resize(x_image_decoded, [self.image_size, self.image_size])
            y_image_resized = tf.image.resize(y_image_decoded, [self.image_size, self.image_size])

            x_image_norm = x_image_resized / 255
            y_image_norm = y_image_resized / 255

            return x_image_norm, y_image_norm

        
        dataset = tf.data.Dataset.zip((x_files, y_files))

        dataset = dataset.interleave(
            lambda x: tf.data.Dataset.from_tensor_slices(x),
            cycle_length=tf.data.AUTOTUNE,
            num_parallel_calls=tf.data.AUTOTUNE
        )

        dataset = dataset.map(read_image, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(1000).batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def read_predict_data(self):
        file_path = '../datasets/test/pre.png'
        image = tf.io.read_file(file_path)
        image_decoded = tf.image.decode_png(image, channels=3)
        image_resized = tf.image.resize(image_decoded, [self.image_size, self.image_size])
        image_norm = image_resized / 255

        return image_norm

    def save_image(self, image):
        file_path = '../datasets/test/result.png'
        image = image * 255

        encode_image = tf.image.encode_jpeg(image, format='rgb', quality=100)

        with open(file_path, 'wb') as fd:
            fd.write(encode_image)


# processor = ImageProcessor()

# train_data = processor.read_train_data()
# predict_data = processor.read_predict_data()

# # Perform training and prediction using the data
# # ...

# processor.save_image(result_image)