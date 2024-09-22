#!/bin/bash

# Predefined list of source folders
SOURCE_FOLDERS=(
    "/home/bouhsi95/seenn/data/modalities/normal/ground_train"
    "/home/bouhsi95/seenn/data/modalities/normal/in_flight"
    "/home/bouhsi95/seenn/data/modalities/normal/inflight_train"
    "/home/bouhsi95/seenn/data/modalities/normal/no_clouds"
    "/home/bouhsi95/seenn/data/modalities/normal/overcast_cumulus"
    # Add more source folders as needed
)

# Predefined destination folder
DESTINATION_FOLDER="/home/bouhsi95/seenn/data/experiments/normal"

# The path to the Python script
PYTHON_SCRIPT_PATH="/home/bouhsi95/seenn/src/utils/create_exp_data.py"

# Call the Python script with the predefined destination folder and the source folders
nohup python "$PYTHON_SCRIPT_PATH" "$DESTINATION_FOLDER" "${SOURCE_FOLDERS[@]}" > exp_data.out

