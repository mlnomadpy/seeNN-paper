import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, GlobalAveragePooling2D, Dropout, Input, Multiply, LayerNormalization, MultiHeadAttention, Concatenate
from tensorflow.keras.models import Model
from utils.dcp_layer import DarkChannelPriorLayerV2
from utils.edge_layer import EdgeDetectionLayerV2
from utils.densenet121 import densenet
from utils.self_entropy import SelfEntropyLayerV2

def project_embeddings(
    embeddings, num_projection_layers, projection_dims, dropout_rate
):
    projected_embeddings = tf.keras.layers.Dense(units=projection_dims)(embeddings)
    for _ in range(num_projection_layers):
        x = tf.nn.gelu(projected_embeddings)
        x = tf.keras.layers.Dense(projection_dims)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        x = tf.keras.layers.Add()([projected_embeddings, x])
        projected_embeddings = tf.keras.layers.LayerNormalization()(x)
    return projected_embeddings

def build_model(config):
    num_channels = 3
    number_of_modalities = 0
    edge_layer = config.edge  # set it to true if you want to use EdgeDetectionLayer
    entropy_layer = config.entropy
    dark_channel_layer = config.dcp
    rgb_layer = config.rgb
    depth_layer = config.depth
    normal_layer = config.normal

    inputs = Input(shape=(config.img_height, config.img_width, num_channels))
    
    
    x = inputs
    embeddings = []

    x_edge = None
    x_depth = None
    x_normal = None
    x_entropy = None
    x_dark_channel = None
    x_rgb = None

    base_model_rgb = None
    base_model_edge = None
    base_model_entropy = None
    base_model_dark_channel = None
    base_model_depth = None
    base_model_normal = None

    # Define base models for each modality
    if rgb_layer:
        base_model_rgb = densenet(input_shape=(config.img_height, config.img_width, num_channels))
        base_model_rgb.trainable = config.trainable_epochs == 0
        x_rgb = base_model_rgb(x)
        x_rgb = GlobalAveragePooling2D()(x_rgb)
        rgb_proj_head = project_embeddings( x_rgb, config.num_projection_layers, config.projection_dims, config.proj_dropout_rate)

        embeddings.append(rgb_proj_head)
        number_of_modalities += 1

    if edge_layer:
        base_model_edge = densenet(input_shape=(config.img_height, config.img_width, num_channels))
        base_model_edge.trainable = config.trainable_epochs == 0
        x_edge = EdgeDetectionLayerV2()(inputs)
        x_edge = base_model_edge(x_edge)
        x_edge = GlobalAveragePooling2D()(x_edge)
        edge_proj_head = project_embeddings( x_edge, config.num_projection_layers, config.projection_dims, config.proj_dropout_rate)
 
        embeddings.append(edge_proj_head)
        number_of_modalities += 1

    if entropy_layer:
        base_model_entropy = densenet(input_shape=(config.img_height, config.img_width, num_channels))
        base_model_entropy.trainable = config.trainable_epochs == 0
        x_entropy = SelfEntropyLayerV2()(inputs)
        x_entropy = base_model_entropy(x_entropy)
        x_entropy = GlobalAveragePooling2D()(x_entropy)
        entropy_proj_head = project_embeddings( x_entropy, config.num_projection_layers, config.projection_dims, config.proj_dropout_rate)

        embeddings.append(entropy_proj_head)
        number_of_modalities += 1

    if dark_channel_layer:
        base_model_dark_channel = densenet(input_shape=(config.img_height, config.img_width, num_channels))
        base_model_dark_channel.trainable = config.trainable_epochs == 0
        x_dark_channel = DarkChannelPriorLayerV2()(inputs)
        x_dark_channel = base_model_dark_channel(x_dark_channel)
        x_dark_channel = GlobalAveragePooling2D()(x_dark_channel)
        entropy_proj_head = project_embeddings( x_dark_channel, config.num_projection_layers, config.projection_dims, config.proj_dropout_rate)

        embeddings.append(x_dark_channel)
        number_of_modalities += 1

    if depth_layer:
        depth_inputs = Input(shape=(config.img_height, config.img_width, num_channels))
        base_model_depth = densenet(input_shape=(config.img_height, config.img_width, num_channels))
        base_model_depth.trainable = config.trainable_epochs == 0
        x_depth = base_model_depth(depth_inputs)
        x_depth = GlobalAveragePooling2D()(x_depth)

        depth_proj_head = project_embeddings( x_depth, config.num_projection_layers, config.projection_dims, config.proj_dropout_rate)

        embeddings.append(depth_proj_head)
        number_of_modalities += 1



    if normal_layer:
        normal_inputs = Input(shape=(config.img_height, config.img_width, num_channels))
        base_model_normal = densenet(input_shape=(config.img_height, config.img_width, num_channels))
        base_model_normal.trainable = config.trainable_epochs == 0
        x_normal = base_model_normal(normal_inputs)
        x_normal = GlobalAveragePooling2D()(x_normal)
        normal_proj_head = project_embeddings( x_normal, config.num_projection_layers, config.projection_dims, config.proj_dropout_rate)
        embeddings.append(normal_proj_head)
        number_of_modalities += 1

    # Concatenate normalized embeddings for weight generation
    concatenated_embeddings = Concatenate(axis=-1)(embeddings)

    # Weighted Fusion
    if config.apply_weights:
        # Learnable Weight Vector
        weights_vector = Dense(len(embeddings), activation='sigmoid')(concatenated_embeddings)

        # Apply learned weights to each modality embedding
        weighted_embeddings = [Multiply()([embedding, weights_vector[:, i:i+1]]) for i, embedding in enumerate(concatenated_embeddings)]
        xc = Concatenate(axis=1)(weighted_embeddings)  # Fusion of weighted embeddings
    else:
        xc = concatenated_embeddings  # For Summation without weights

    xc_shape = xc.shape

    xc = tf.reshape(concatenated_embeddings, [-1, number_of_modalities, embeddings[0].shape[-1]])
    print(xc.shape)

    feature_dim = xc_shape[-1]  # xc is of shape [batch_size, sequence_length, feature_dim]

    # Choose num_heads such that it divides feature_dim
    if feature_dim % config.num_heads != 0:
        raise ValueError("feature_dim must be divisible by num_heads")

    # key_dim = feature_dim // num_heads
    if config.fusion_type == "attention":
        transformer_layer = MultiHeadAttention(num_heads=config.num_heads, key_dim=config.key_dim)
        # Self-Attention
        xc_transformer = transformer_layer(xc, xc)
        xc = LayerNormalization()(xc_transformer)
    
    

    # Flattening
    xc_transformer = Flatten()(xc)  # Flatten 

    # Dense layers
    xc_transformer = Dense(256, activation='relu')(xc_transformer)
    xc_transformer = Dropout(rate=0.2)(xc_transformer)
    xc_transformer = Dense(256, activation='relu')(xc_transformer)
    xc_transformer = Dropout(rate=0.2)(xc_transformer)

    predictions = Dense(config.num_classes, activation='softmax')(xc_transformer)

    model_inputs = []
    if rgb_layer or edge_layer or entropy_layer:
        model_inputs.append(inputs)
    if depth_layer:
        model_inputs.append(depth_inputs)
    if normal_layer:
        model_inputs.append(normal_inputs)

    model = Model(inputs=model_inputs, outputs=predictions)
    return model
