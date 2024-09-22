import tensorflow as tf
from tensorflow.keras import layers


class SelfEntropyLayerV2(layers.Layer):
    def __init__(self, **kwargs):
        super(SelfEntropyLayerV2, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SelfEntropyLayerV2, self).build(input_shape)
    def call(self, inputs):
        def process_channel(channel_image):
            # Pad the channel image
            img_padded = tf.pad(channel_image, paddings=[[0, 0], [4, 4], [4, 4], [0, 0]])

            # Calculate entropy patches and reshape
            patches = tf.image.extract_patches(
                images=img_padded,
                sizes=[1, 9, 9, 1],
                strides=[1, 1, 1, 1],
                rates=[1, 1, 1, 1],
                padding='VALID'
            )
            entropy = -tf.reduce_sum(patches * tf.math.log(patches + 1e-10) / tf.math.log(2.0), axis=3)
            resized_entropy = tf.reshape(entropy, [-1, inputs.shape[1], inputs.shape[2], 1])

            return resized_entropy

        # Split the input tensor into its separate channels
        channels = tf.split(inputs, num_or_size_splits=3, axis=3)

        # Process each channel and store the results
        processed_channels = [process_channel(channel) for channel in channels]

        # Concatenate the processed channels along the channel dimension
        entropy_output = tf.concat(processed_channels, axis=3)

        return entropy_output

    # def call(self, inputs):
    #     # Pad each channel of the input
    #     def process_channel(channel_image):
    #         # Add a batch dimension to channel_image
    #         channel_image = tf.expand_dims(channel_image, axis=0)

    #         img_padded = tf.pad(channel_image, paddings=[[0, 0], [4, 4], [4, 4], [0, 0]])
    #         patches = tf.image.extract_patches(
    #             images=img_padded,
    #             sizes=[1, 9, 9, 1],
    #             strides=[1, 1, 1, 1],
    #             rates=[1, 1, 1, 1],
    #             padding='VALID'
    #         )
    #         entropy = -tf.reduce_sum(patches * tf.math.log(patches + 1e-10) / tf.math.log(2.0), axis=3)
    #         resized_entropy = tf.reshape(entropy, [-1, inputs.shape[1], inputs.shape[2], 1])
    #         return resized_entropy

    #         # Split the channels and process each one
    #     channels = tf.split(inputs, num_or_size_splits=3, axis=3)
    #     processed_channels = [process_channel(channel) for channel in channels]
    #     entropy_output = tf.concat(processed_channels, axis=3)
    #     return entropy_output
