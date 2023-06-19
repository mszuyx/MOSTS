# MOSTS

This repo is the official implementation of **A Self-Supervised Miniature One-Shot Texture Segmentation (MOSTS) Model for Real-Time Robot Navigation and Embedded Applications**.

## To run model training on your own pc:

Navigate to the folder directory, open a terminal and create a virtual environment:
```
python3 -m venv env               # Create a virtual environment

source env/bin/activate           # Activate virtual environment
```
Install none pytorch dependencies:
```
pip install -r requirements.txt   # Install dependencies
```
Install pytorch 1.10 (you might have to use a different version depending on your CUDA version)
```
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
Install albumentations if you want to play with data augmentation:
```
pip install -U albumentations
```
To start the training:
```
python3 train_ablation.py
```
**If you encounter some runtime memory issues, you can decrease the batch_size / num_workers according to your GPU spec**
**Remember to change the dataset file path in "ablation_data_loader" according to your file system.**

To exit virtual environment:
```
deactivate                       # Exit virtual environment
```

## DTD dataset can be downloaded from here:
https://www.robots.ox.ac.uk/~vgg/data/dtd/

## The encoder backbone (pre-trained moblienetv3) can be downloaded from here:

Please place the downloaded .pth file under /utils/model/ for the train_ablation.py to work.
