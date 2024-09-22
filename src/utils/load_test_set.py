from keras.preprocessing.image import ImageDataGenerator
import os
import pandas as pd
import random

def limit_data(data_dir, n=10000):
    a = []
    
    files_list = os.listdir(data_dir)
    random.shuffle(files_list)

    for i in files_list:
        class_files = os.listdir(os.path.join(data_dir, i))
        random.shuffle(class_files)
        for file in class_files[:n]:
            a.append((os.path.join(data_dir, i, file), i))
    return pd.DataFrame(a, columns=['filename', 'class'])

def load_test_data(config):
    # Create a basic ImageDataGenerator for the test set
    datagen = ImageDataGenerator(rescale=1./255)

    # Apply undersampling method to limit the data
    test_df = limit_data(config.test_dataset_path, 1000)

    # Create the test data generator
    test_generator = datagen.flow_from_dataframe(
        test_df,
        x_col='filename',
        y_col='class',
        target_size=(config.img_height, config.img_width),
        batch_size=config.batch_size,
        class_mode='categorical',
        shuffle=False)  # No need to shuffle the test data
    print("Test data loaded")

    return test_generator
