<h1 align="center"> AdRecreate: A project - recreating advertisements. </h1> 
<p align="center">
  <a href="#Introduction">Introduction</a> |
  <a href="#Usage">Usage</a> |
</p >

## Introduction

Welcome to our project, aiming to redefine digital advertising using AI and Deep Learning. We focus on removing text from ads, detecting font styles, and recreating advertisements for enhanced visual impact and diversity.

At the core of our project are three technologies: DPText-DETR, Unet with a Resnet50 backbone, and a custom font detection model trained on 60 distinct fonts.

DPText-DETR is a powerful text detection transformer, designed to predict polygon points for localizing text. This method ensures high training efficiency, robustness, and state-of-the-art performance in text localization.

Unet with a Resnet50 backbone is utilized for the semantic segmentation of images, facilitating efficient identification and extraction of text regions.

Our font detection model, trained on a wide array of 60 fonts, aids in identifying specific fonts used in ads, which is crucial for the redesign process.

Through these combined methodologies, our project promises effective text removal, precise font detection, and the recreation of engaging and diverse advertisements. We're thrilled to share this project with developers, researchers, marketers, and AI enthusiasts and look forward to its impact on the world of digital advertising.

## Usage

It's recommended to configure the environment using Anaconda. Python 3.8 + PyTorch 1.9.1 (or 1.9.0) + CUDA 11.1 + Detectron2 (v0.6) are suggested.

- ### Installation
```
conda create -n DPText-DETR python=3.8 -y
conda activate DPText-DETR
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install opencv-python scipy timm shapely albumentations Polygon3
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
pip install setuptools==59.5.0
git clone https://github.com/ymy-k/DPText-DETR.git
cd DPText-DETR
python setup.py build develop
```

- ### Models
Download the models and keep them in `weights` folder.
https://drive.google.com/drive/folders/16zdy60uvwTmVrLVctR4fvAaECocPJO9Y?usp=share_link

The generation of positional label form is provided in `process_positional_label.py`
