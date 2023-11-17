# Miniature One-Shot Texture Segmentation (MOSTS) 

This repo is the official implementation of **A Self-Supervised Miniature One-Shot Texture Segmentation (MOSTS) Model for Real-Time Robot Navigation and Embedded Applications**.
[https://arxiv.org/pdf/2306.08814v1.pdf]
```bib
@article{chen2023self,
  title={A Self-Supervised Miniature One-Shot Texture Segmentation (MOSTS) Model for Real-Time Robot Navigation and Embedded Applications},
  author={Chen, Yu and Rastogi, Chirag and Zhou, Zheyu and Norris, William R},
  journal={arXiv preprint arXiv:2306.08814},
  year={2023}
}
```
This repo is under the Creative Commons Attribution-NonCommercial-ShareAlike license. (**CC-BY-NC-SA**)

Please cite our work if you find this repo helpful! : )

## MOSTS system architecture:
![Picture4 (1)](https://github.com/mszuyx/MOSTS/assets/37651144/9b05a6b4-9c7e-4652-86df-f025d31a4a57)

## Overall system flow chart
![Picture3](https://github.com/mszuyx/MOSTS/assets/37651144/60a06c3e-aa04-4fa5-b64e-e615784b5ef3)

## Example results
![Picture2](https://github.com/mszuyx/MOSTS/assets/37651144/0c282189-93fd-4bda-a76a-181cd88f3743)

## Demo video
https://github.com/mszuyx/MOSTS/assets/37651144/44ce340f-95b6-4923-af14-cd3f2f1cda7f

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

## The encoder backbone (pretrained/mobilenetv3-large-1cd25616.pth) can be downloaded from here:
https://github.com/d-li14/mobilenetv3.pytorch
Please place the downloaded .pth file under /utils/model/ for the train_ablation.py to work.

## DTD dataset can be downloaded from here:
https://www.robots.ox.ac.uk/~vgg/data/dtd/

## The Idoor Small Object Dataset (ISOD) can be downloaded from here:
https://www.kaggle.com/datasets/yuchen66/indoor-small-object-dataset
![Picture1](https://github.com/mszuyx/MOSTS/assets/37651144/8150d327-7231-4fd3-bf0a-928a4ccfe36a)

Please cite our work if you want to use this dataset for research/publication!

