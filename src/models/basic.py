from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

def create_model(input_shape, num_classes):
    model = Sequential([
        # Convolutional layer 1
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Convolutional layer 2
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Convolutional layer 3
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Flattening the output of the convolutions
        Flatten(),
        
        # Dense layer 1
        Dense(128, activation='relu'),
        Dropout(0.5),
        
        # Output layer
        Dense(num_classes, activation='softmax')
    ])

    return model
