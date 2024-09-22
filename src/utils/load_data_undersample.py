from keras.preprocessing.image import ImageDataGenerator
import os
import pandas as pd

import random


def limit_data(data_dir, n=100000):
    a = []
    
    files_list = os.listdir(data_dir)
    random.shuffle(files_list)

    for i in files_list:
        for k, j in enumerate(os.listdir(data_dir+'/'+i)):
            if k > n:
                continue
            a.append((f'{data_dir}/{i}/{j}', i))
    return pd.DataFrame(a, columns=['filename', 'class'])


def load_data(config):
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=config.val_split)  # set validation split
    df = limit_data(config.dataset_path, config.num_img_lim)
    # Add a filter to df to create train, validation, test dfs
    # 1.83304957311693_6N7 Sealane 01 85deg_500agl_0_rgb_2.png
    # 2.9825891359190724_KMDW Runway 13C_0agl_0_rgb_2
    validation_views = ['KSFO Runway 19L', 'KLAX Runway 24R 19deg', 'KACY Runway 31 19deg', 'CYQB Runway 29 252deg', '6N7 Sealane 01 146deg', 'KLGB Runway 08L 146deg']
    
    df_train = df[~df['filename'].str.contains('|'.join(validation_views))]
    df_validation = df[df['filename'].str.contains('|'.join(validation_views))]

    train_generator = datagen.flow_from_dataframe(
        df_train,
        x_col='filename',
        y_col='class',
        seed=config.seed,
        target_size=(config.img_height, config.img_width),
        batch_size=config.batch_size,
        class_mode='categorical',
        # subset='training',
        shuffle=True)  # set as training data

    validation_generator = datagen.flow_from_dataframe(
        df_validation,
        x_col='filename',
        y_col='class',
        target_size=(config.img_height, config.img_width),
        batch_size=config.batch_size,
        class_mode='categorical')  # set as validation data

    return train_generator, validation_generator