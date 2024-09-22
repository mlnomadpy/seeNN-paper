from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence
import os
import pandas as pd
import random
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import numpy as np

def limit_data(config, n=100000):
    a = []
    
    rgb_dir = config.dataset_path
    depth_dir = config.depth_path
    normal_dir = config.normal_path

    print(f"RGB Directory: {rgb_dir}")
    print(f"Depth Directory: {depth_dir}")
    print(f"Normal Directory: {normal_dir}")

    files_list = os.listdir(rgb_dir)
    random.shuffle(files_list)

    for folder_name in files_list:
        rgb_folder = os.path.join(rgb_dir, folder_name)
        depth_folder = os.path.join(depth_dir, folder_name)
        normal_folder = os.path.join(normal_dir, folder_name)

        if not os.path.isdir(rgb_folder) or not os.path.isdir(depth_folder) or not os.path.isdir(normal_folder):
            print(f"Missing folder for class '{folder_name}'. Check RGB, Depth, and Normal directories.")
            continue

        rgb_files = os.listdir(rgb_folder)
        for k, rgb_file in enumerate(rgb_files):
            if k >= n:
                break

            base_name = rgb_file.split('_rgb_')[0]
            depth_file_name = f"{base_name}_depth_{rgb_file.split('_rgb_')[1]}"
            normal_file_name = f"{base_name}_normal_{rgb_file.split('_rgb_')[1]}"

            rgb_path = os.path.join(rgb_folder, rgb_file)
            depth_path = os.path.join(depth_folder, depth_file_name)
            normal_path = os.path.join(normal_folder, normal_file_name)

            if os.path.exists(depth_path) and os.path.exists(normal_path):
                a.append((rgb_path, depth_path, normal_path, folder_name))
            else:
                if not os.path.exists(depth_path):
                    print(f"Depth file not found: {depth_path}")
                if not os.path.exists(normal_path):
                    print(f"Normal file not found: {normal_path}")

    df = pd.DataFrame(a, columns=['rgb', 'depth', 'normal', 'class'])
    print(f"Total image triplets found: {len(df)}")
    return df


class MultiModalDataGenerator(Sequence):
    def __init__(self, df, config, subset):
        self.df = df
        self.batch_size = config.batch_size
        self.target_size = (config.img_height, config.img_width)
        self.subset = subset
        self.config = config
        validation_views = ['KSFO Runway 19L', 'KLAX Runway 24R 19deg', 'KACY Runway 31 19deg', 'CYQB Runway 29 252deg', '6N7 Sealane 01 146deg', 'KLGB Runway 08L 146deg']
    
        if subset == 'training':
            df_train = self.df[~self.df['rgb'].str.contains('|'.join(validation_views))]
            self.df = df_train
            # self.df = self.df.sample(frac=1-config.val_split, random_state=config.seed)
        elif subset == 'validation':
            df_validation = self.df[self.df['rgb'].str.contains('|'.join(validation_views))]
            self.df = df_validation

            # self.df = self.df.drop(self.df.sample(frac=1-config.val_split, random_state=config.seed).index)


        self.log_image_counts()

    def __len__(self):
        return int(np.ceil(len(self.df) / float(self.batch_size)))
    def log_image_counts(self):
        self.num_images = len(self.df)
        counts_per_set = self.df['class'].value_counts()
        print(f"Total image sets in {self.subset} set:")
        print(counts_per_set)

    def __getitem__(self, idx):
        batch_df = self.df.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]

        rgb_images = []
        depth_images = []
        normal_images = []

        for _, row in batch_df.iterrows():
            if self.config.rgb or self.config.edge or self.config.entropy:
                rgb_img = tf.io.read_file(row['rgb'])
                rgb_img = tf.image.decode_png(rgb_img, channels=3)
                rgb_img = tf.image.resize(rgb_img, self.target_size)
                rgb_images.append(rgb_img)
            if self.config.depth:            
                depth_img = tf.io.read_file(row['depth'])
                depth_img = tf.image.decode_png(depth_img, channels=3)
                depth_img = tf.image.resize(depth_img, self.target_size)
                depth_images.append(depth_img)
            if self.config.normal:            
                normal_img = tf.io.read_file(row['normal'])
                normal_img = tf.image.decode_png(normal_img, channels=3)
                normal_img = tf.image.resize(normal_img, self.target_size)
                normal_images.append(normal_img)

        inputs = []
        # Normalize the images
        if self.config.rgb or self.config.edge or self.config.entropy:
            rgb_images = tf.stack(rgb_images) / 255.0
            inputs.append(rgb_images)
        if self.config.depth:
            depth_images = tf.stack(depth_images) / 255.0
            inputs.append(depth_images)
        if self.config.normal:
            normal_images = tf.stack(normal_images) / 255.0
            inputs.append(normal_images)


        # Targets (class labels)
        # Assuming class labels are strings like 'bin_0', 'bin_1', ..., 'bin_4'
        class_labels = {'bin_0': 0, 'bin_1': 1, 'bin_2': 2, 'bin_3': 3, 'bin_4': 4}
        
        # Convert class labels to integers
        int_labels = batch_df['class'].replace(class_labels).values

        # Convert integer labels to one-hot encoding
        targets = to_categorical(int_labels, num_classes=self.config.num_classes)  # specify the total number of classes

        return inputs, targets

    def reset(self):
        self.df = self.df.sample(frac=1) if self.subset == 'training' else self.df
        self.on_epoch_end()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.df))
        if self.subset == 'training':
            np.random.shuffle(self.indexes)


def load_data(config):
    full_df = limit_data(config, config.num_img_lim)

    train_generator = MultiModalDataGenerator(full_df, config, subset='training')
    validation_generator = MultiModalDataGenerator(full_df, config, subset='validation')

    return train_generator, validation_generator
