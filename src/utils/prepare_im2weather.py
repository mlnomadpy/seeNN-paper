import os
import json
from PIL import Image
import concurrent.futures
import sys

import os
import pandas as pd
from PIL import Image
import concurrent.futures
import sys
from ast import literal_eval

def process_images(measure_type, bins, bounding_box, source_directory, destination_directory, json_file):
    # Load visibility data into a pandas DataFrame
    df = pd.read_json(json_file)

    df = pd.read_csv(json_file, converters={'loc': literal_eval, 'weather': literal_eval})
    # Flatten the 'loc' column
    loc_df = pd.json_normalize(df['loc'])

    # Since 'weather' is more complex and nested, handle it separately
    weather_df = pd.json_normalize(df['weather'])

    # For nested structures like 'utcdate' within 'weather', you might need to flatten it separately and then concatenate it back
    utcdate_df = pd.json_normalize(df['weather'].apply(lambda x: x.get('utcdate', {})))
    utcdate_df.columns = ['utcdate_' + col for col in utcdate_df.columns]  # Prefix columns for clarity

    # Concatenate the flattened 'loc', 'weather', and 'utcdate' DataFrames horizontally with the original 'elevation' and 'id'
    final_df = pd.concat([df[['elevation', 'id']], loc_df, weather_df.drop(columns=['utcdate']), utcdate_df], axis=1)

    final_df.to_csv('metadata.csv', index=False)
    df = final_df
    # Create a column for visibility if your JSON structure is nested
    # Adjust this line if your JSON structure differs
    df['vis_measure'] = df.apply(lambda row: row['weather']['visi'], axis=1)

    tasks = []
    for image_name in os.listdir(source_directory):
        image_path = os.path.join(source_directory, image_name)
        if os.path.isfile(image_path):
            # Extract the id from the image name
            image_id = extract_id_from_name(image_name)
            
            # Use DataFrame to get visibility value
            if image_id in df['id'].values:
                vis_measure = df.loc[df['id'] == image_id, 'vis_measure'].iloc[0]
                tasks.append((image_path, measure_type, bins, bounding_box, destination_directory, vis_measure))

    # Process images in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(process_single_image, tasks)

def extract_id_from_name(image_name):
    # Extract the id from the image name (implement this based on your naming convention)
    # Example: if image name is 'image_331542624.png', return '331542624'
    return image_name.split('_')[1].split('.')[0]

def process_single_image(task):
    image_path, measure_type, bins, bounding_box, destination_directory, vis_measure = task

    # Determine the bin
    bin_index = determine_bin(float(vis_measure), measure_type, bins)
    bin_dir = os.path.join(destination_directory, f'bin_{bin_index}')
    os.makedirs(bin_dir, exist_ok=True)

    # Process and save the image
    new_image_name = os.path.basename(image_path)
    process_and_save_image(image_path, bin_dir, new_image_name, bounding_box)

def determine_bin(vis_value, measure_type, bins):
    for i, bin_upper_limit in enumerate(bins):
        if vis_value <= bin_upper_limit:
            return i
    return len(bins)

def process_and_save_image(image_path, bin_dir, new_image_name, bounding_box):
    with Image.open(image_path) as img:
        cropped_img = img.crop(bounding_box)
        cropped_img.save(os.path.join(bin_dir, new_image_name))

if __name__ == "__main__":
    # Checking if the required arguments are provided
    if len(sys.argv) != 7:
        print("Usage: python script.py [measure_type] [bins] [bounding_box] [source_directory] [destination_directory] [json_file]")
        sys.exit(1)

    # Assign command line arguments to variables
    measure_type = sys.argv[1]
    bins = [float(bin_str) for bin_str in sys.argv[2].split(',')]
    bounding_box = tuple(map(int, sys.argv[3].split(',')))
    source_directory = sys.argv[4]
    destination_directory = sys.argv[5]
    json_file = sys.argv[6]

    # Call the function with the provided parameters
    process_images(measure_type, bins, bounding_box, source_directory, destination_directory, json_file)
