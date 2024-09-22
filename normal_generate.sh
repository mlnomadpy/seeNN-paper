# conda init
# conda create -n omnidata -y python=3.8
# conda activate omnidata

# pip install -r /home/bouhsi95/seenn/omnidata/omnidata_tools/torch/requirements.txt


PATH_TO_IMAGE_OR_FOLDER= '/home/bouhsi95/seenn/data/processed/ground_train/bin_0'
PATH_TO_SAVE_OUTPUT='/home/bouhsi95/seenn/data/extracted_features/ground_train/bin_0'

nohup python3 /home/bouhsi95/seenn/omnidata/omnidata_tools/torch/batch_demo.py --task depth --output_path $PATH_TO_SAVE_OUTPUT > depth.out    