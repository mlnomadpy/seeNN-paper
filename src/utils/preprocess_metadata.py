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
    # df = pd.read_json(json_file)

    df = pd.read_csv(json_file)
    print('normalizing loc')
    # Flatten the 'loc' column
    loc_df = pd.json_normalize(df['loc'])
    print('normalizing weather')

    # Since 'weather' is more complex and nested, handle it separately
    weather_df = pd.json_normalize(df['weather'])

    # For nested structures like 'utcdate' within 'weather', you might need to flatten it separately and then concatenate it back
    utcdate_df = pd.json_normalize(df['weather'].apply(lambda x: x.get('utcdate', {})))
    utcdate_df.columns = ['utcdate_' + col for col in utcdate_df.columns]  # Prefix columns for clarity

    # Concatenate the flattened 'loc', 'weather', and 'utcdate' DataFrames horizontally with the original 'elevation' and 'id'
    final_df = pd.concat([df[['elevation', 'id']], loc_df, weather_df.drop(columns=['utcdate']), utcdate_df], axis=1)

    final_df.to_csv('processed_metadata.csv', index=False)


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
