import tensorflow as tf
from tensorflow.keras import layers


class DarkChannelPriorLayerV2(layers.Layer):
    def __init__(self, window_size=15, **kwargs):
        self.window_size = window_size
        super(DarkChannelPriorLayerV2, self).__init__(**kwargs)

    def build(self, input_shape):
        super(DarkChannelPriorLayerV2, self).build(input_shape)

    def call(self, inputs):
        # Function to apply on each channel
        def process_channel(channel_image):
            # Add a batch dimension to channel_image
            channel_image = tf.expand_dims(channel_image, axis=-1)

            patches = tf.image.extract_patches(
                images=channel_image,
                sizes=[1, self.window_size, self.window_size, 1],
                strides=[1, 1, 1, 1],
                rates=[1, 1, 1, 1],
                padding='SAME'
            )
            dark_channel_prior = tf.reduce_min(patches, axis=-1)
            return dark_channel_prior

        # Reshape input to separate channels and apply tf.map_fn
        reshaped_input = tf.transpose(inputs, [3, 0, 1, 2])
        dark_channel_prior_per_channel = tf.map_fn(process_channel, reshaped_input, dtype=tf.float32)

        # Revert back to the original shape
        dark_channel_prior = tf.transpose(dark_channel_prior_per_channel, [1, 2, 3, 0])

        return dark_channel_prior