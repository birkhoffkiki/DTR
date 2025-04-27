<!-- # DTR
##  -->
![header](https://capsule-render.vercel.app/api?type=waving&height=200&color=gradient&text=DGR&desc=Generative%20AI%20for%20Misalignment-Resistant%20Virtual%20%20Staining%20to%20Accelerate%20Histopathology%20Workflows&descSize=19&fontAlign=8&fontAlignY=19&animation=twinkling&fontSize=50&descAlignY=43)
![GitHub last commit](https://img.shields.io/github/last-commit/birkhoffkiki/DTR?style=flat-square)
[![X (formerly Twitter) Follow](https://img.shields.io/twitter/follow/SMARTLab_HKUST%20)](https://x.com/SMARTLab_HKUST)
![Python Version](https://img.shields.io/badge/python-3.9-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)

--- 
**Last Updated**: 27/04/2025

The official implementation of DGR, a generative AI model for virtual staining in histopathology workflows.

![main_figure](assets/main.png)

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Training](#training)
- [Pretrained Models](#pretrained-models)
- [Inference](#inference)
- [Results](#results)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview
DGR is a novel  framework designed for virtual staining of histopathology images with enhanced resistance to misalignment. Our method enables:
- High-fidelity stain transformation between different histopathology modalities
- Robust performance despite common tissue section misalignments
- Significant acceleration of histopathology workflows

## Key Features
- 🚀 **High-quality transformations**: Preserves tissue structures while converting stains
- 🔄 **Misalignment-resistant**: Handles up to 50% tissue section offset
- ⏱️ **Fast inference**: Processes whole-slide images in minutes
- 📊 **Multi-dataset support**: Compatible with Aperio-Hamamatsu and HEMIT datasets
- 🧠 **Modular architecture**: Easy to extend for new stain types

## Installation

### Setup
1. Clone this repository:
```bash
git clone https://github.com/birkhoffkiki/DTR.git
cd DTR
conda create --name DTR python=3.9
conda activate DTR
pip install -r requirements.txt
```

### Data preparation

* Aperio-Hamamatsu dataset: https://github.com/khtao/StainNet
* HEMIT dataset: https://github.com/BianChang/HEMIT-DATASET  

### Training
```bash
# For Aperio-Hamamatsu dataset
bash train_aperio.sh

# For HEMIT dataset
bash train_hemit.sh
```
## Pretrained Models

| Model Name       | Download Link |
|------------------|---------------|
| AF2HE Weight     | [Download](https://github.com/birkhoffkiki/DTR/releases/download/weights/af2he_weight.pth) |
| HE2PAS Weight   | [Download](https://github.com/birkhoffkiki/DTR/releases/download/weights/he2pas_weight.pth) |
| HEMIT Weight    | [Download](https://github.com/birkhoffkiki/DTR/releases/download/weights/hemit_weight.pth) |
| Aperio Weight   | [Download](https://github.com/birkhoffkiki/DTR/releases/download/weights/aperio_weight.pth) |


## Inference
Example notebook: `play_with_the_pretrained_model.ipynb`

## contact

if you have any questions, please feel free to contact me:  

* JIABO MA, jmabq@connect.ust.hk
