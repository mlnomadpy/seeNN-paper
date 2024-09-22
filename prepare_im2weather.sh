#!/bin/bash

# Define variables
PYTHON_SCRIPT="/home/bouhsi95/seenn/src/utils/preprocess_metadata.py"
MEASURE_TYPE="miles"  # or "meters", depending on your requirement
BINS="0.5,1,3,5,100"  # The bin ranges
BOUNDING_BOX="20,63,1282,745"  # The bounding box coordinates
SOURCE_DIR="/home/bouhsi95/seenn/data/real_binned/rgb"  # Path to the source directory
DEST_DIR="/home/bouhsi95/seenn/data/real_images_raw"  # Path to the destination directory
JSON_FILE="/home/bouhsi95/seenn/data/real_raw/metadata.json"  # Path to the JSON file with visibility data

# Check if the Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

# Execute the Python script
python "$PYTHON_SCRIPT" "$MEASURE_TYPE" "$BINS" "$BOUNDING_BOX" "$SOURCE_DIR" "$DEST_DIR" "$JSON_FILE"
