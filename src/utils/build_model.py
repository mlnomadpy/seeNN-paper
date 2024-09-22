from tensorflow.keras.applications import DenseNet121, VGG16, ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from utils.dcp_layer import DarkChannelPriorLayerV2
from utils.edge_layer import EdgeDetectionLayerV2
from utils.densenet121 import densenet
from utils.self_entropy import SelfEntropyLayerV2
import tensorflow as tf


def build_model(config):
    num_channels = 3

    edge_layer = config.edge  # set it to true if you want to use EdgeDetectionLayer
    entropy_layer = config.entropy
    dark_channel_layer = config.dcp
    rgb_layer = config.rgb


    inputs = Input(shape=( config.img_height, config.img_width, num_channels))
    x = inputs

    edge_output = None
    entropy_output = None
    dark_channel_output = None

    x_edge = None
    x_entropy = None
    x_dark_channel = None
    x_rgb = None

    base_model_rgb = None
    base_model_edge = None
    base_model_entropy = None
    base_model_dark_channel = None

    # for the first epochs the backbone will not be trainable
    trainable = (config.trainable_epochs == 0)
    print(f'Trainable: {trainable}')
    embeddings = []

    if rgb_layer:
        base_model_rgb = densenet(input_shape=(
            config.img_height, config.img_width, num_channels))
        base_model_rgb.trainable = trainable
        x_rgb = base_model_rgb(x)
        x_rgb = GlobalAveragePooling2D()(x_rgb)
        x_rgb = tf.keras.layers.LayerNormalization()(x_rgb)

        print(x_rgb.shape)
        embeddings.append(x_rgb)

    if edge_layer:
        base_model_edge = densenet(input_shape=(
            config.img_height, config.img_width, num_channels))
        base_model_edge.trainable = trainable

        edge_output = EdgeDetectionLayerV2()(inputs)
        x_edge = edge_output

        x_edge = base_model_edge(x_edge)
        x_edge = GlobalAveragePooling2D()(x_edge)
        x_edge = tf.keras.layers.LayerNormalization()(x_edge)

        print(x_edge.shape)

        embeddings.append(x_edge)

    if entropy_layer:
        base_model_entropy = densenet(input_shape=(
            config.img_height, config.img_width, num_channels))
        base_model_entropy.trainable = trainable

        entropy_output = SelfEntropyLayerV2()(inputs)
        x_entropy = entropy_output

        x_entropy = base_model_entropy(x_entropy)
        x_entropy = GlobalAveragePooling2D()(x_entropy)
        x_entropy = tf.keras.layers.LayerNormalization()(x_entropy)

        print(x_entropy.shape)

        embeddings.append(x_entropy)

    if dark_channel_layer:
        base_model_dark_channel = densenet(input_shape=(
            config.img_height, config.img_width, num_channels))
        base_model_dark_channel.trainable = trainable

        dark_channel_output = DarkChannelPriorLayerV2()(inputs)
        x_dark_channel = dark_channel_output
        x_dark_channel = base_model_dark_channel(x_dark_channel)
        x_dark_channel = GlobalAveragePooling2D()(x_dark_channel)
        x_dark_channel = tf.keras.layers.LayerNormalization()(x_dark_channel)
        print(x_dark_channel.shape)

        embeddings.append(x_dark_channel)

    # Concatenate
    # xc = tf.concat(embeddings, axis=1)
    for l in embeddings:
        print('Embeddings Shapes:')
        print(l.shape)
    # or Add and normalization
    
    # xc = x_rgb + x_dark_channel + x_edge
    xc = tf.add_n(embeddings)
    # Using LayerNormalization for normalization
    # xc = tf.keras.layers.LayerNormalization()(xc)

    xc = Dense(256, activation='relu')(xc)
    xc = Dropout(rate=0.2)(xc)
    xc = Dense(256, activation='relu')(xc)
    xc = Dropout(rate=0.2)(xc)

    predictions = Dense(config.num_classes, activation='softmax')(xc)

    model = Model(inputs=inputs, outputs=predictions)
    print('Model Created')
    return model