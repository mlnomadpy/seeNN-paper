import os
import shutil

def copy_images(source_folder, destination_folder):
    # Check if the destination folder exists, if not, create it
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Walk through the source folder
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            # Check for image file extensions
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')):
                # Construct full file path
                file_path = os.path.join(root, file)
                # Copy file to destination folder
                shutil.copy(file_path, destination_folder)
                print(f"Copied {file} to {destination_folder}")

# Example usage
source_folder = '/home/bouhsi95/seenn/Image'  # Replace with your source folder path
destination_folder = '/home/bouhsi95/seenn/data/real_images_raw'  # Replace with your destination folder path

copy_images(source_folder, destination_folder)
