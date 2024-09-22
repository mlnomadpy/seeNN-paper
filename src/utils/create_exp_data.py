import os
import shutil
import sys

def copy_files(src_folders, dest_folder):
    for src_folder in src_folders:
        # Walk through the source folder
        for dirpath, dirnames, filenames in os.walk(src_folder):
            # Determine the path to the destination folder
            dest_path = os.path.join(dest_folder, os.path.relpath(dirpath, src_folder))

            # Create the destination folder if it doesn't exist
            if not os.path.exists(dest_path):
                os.makedirs(dest_path)

            # Copy each file to the destination folder
            for file in filenames:
                src_file = os.path.join(dirpath, file)
                dest_file = os.path.join(dest_path, file)
                shutil.copy2(src_file, dest_file)
                print(f"Copied {src_file} to {dest_file}")

if __name__ == "__main__":
    # Check if enough arguments are passed
    if len(sys.argv) < 3:
        print("Usage: python script.py <destination_folder> <source_folder1> <source_folder2> ...")
        sys.exit(1)

    destination = sys.argv[1]
    sources = sys.argv[2:]

    copy_files(sources, destination)
