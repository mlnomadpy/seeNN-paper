# SeeNN: Leveraging Multimodal Deep Learning for In-Flight Long-Range Atmospheric Visibility Estimation in Aviation Safety
Taha Bouhsine, Giuseppina Carannant, Nidhal C. Bouaynaya, Soufiane Idbraim, Phuong Tran, Grant Morfit, Maggie Mayfield, Charles Cliff Johnson,


## Overview
SeeNN is a multi-modal image classification system that leverages various image modalities to improve classification accuracy. This project is part of a research paper exploring advanced techniques in computer vision and deep learning.

## Features
- Multi-modal input support (RGB, Depth, Normal, Edge, Entropy)
- Flexible model architecture with attention-based fusion
- Customizable training parameters
- Integration with Weights & Biases for experiment tracking
- Comprehensive evaluation metrics and visualizations

## Requirements
- Python 3.7+
- TensorFlow 2.x
- CUDA-compatible GPU (recommended)
- Additional dependencies listed in `requirements.txt`

## Usage

### Training
To train the model:

```bash
bash train.sh
```


Modify `train_all.sh` to adjust training parameters and enabled modalities.

### Custom Training
For more control, use `train_all.py` directly.

See `train_all.py` for a full list of available arguments.

## Project Structure
- `src/train_all.py`: Main training script
- `src/utils/`: Utility functions for data loading, model building, etc.
- `train_all.sh`: Bash script for easy training execution

## Citation
If you use this code in your research, please cite our paper:


## Contact

contact@tahabouhsine.com
