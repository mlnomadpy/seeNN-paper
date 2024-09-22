import tensorflow as tf

class EdgeDetectionLayerV2(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(EdgeDetectionLayerV2, self).__init__(**kwargs)
    
    def build(self, input_shape):
        super(EdgeDetectionLayerV2, self).build(input_shape)
        
    def call(self, inputs):
        # Perform edge detection on all channels simultaneously
        sobel_edges = tf.image.sobel_edges(inputs)
        
        # Compute the magnitude of the gradient for each channel
        sobel_magnitude = tf.sqrt(tf.reduce_sum(tf.square(sobel_edges), axis=-1))
        
        # Reshape to get the output in the required shape
        sobel_magnitude = tf.reshape(sobel_magnitude, [-1, inputs.shape[1], inputs.shape[2], inputs.shape[3]])
        
        return sobel_magnitude