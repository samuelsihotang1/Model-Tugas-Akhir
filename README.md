# Performance Analysis of Hybrid CNN-ViT Model  
**A Comparative Study of Mobile-Former, CoAtNet, and Modified Models for Image Classification**

## 📌 Description
This repository is part of a final undergraduate project titled **"Performance Analysis of CNN-VIT Hybrid Model with Mobile-Former, CoAtNet, and Modified Models for Image Classification with Various Dataset Sizes."**

The study aims to compare the performance of three deep learning architectures: **Mobile-Former**, **CoAtNet**, and a **Modified Model (Mobile-Former with Relative Attention)** in image classification tasks using both small (Tiny-ImageNet) and large (ImageNet-1K) datasets.

## 🧩 Models Used

- 🔹 **Mobile-Former**  
  An efficient model that combines MobileNet and Transformer architecture with lightweight cross-attention.

- 🔸 **CoAtNet**  
  A model that integrates convolution and self-attention, enhanced with relative attention to capture spatial dependencies.

- 🔧 **Modified Model**  
  A variant of Mobile-Former where multi-head attention is replaced with CoAtNet's relative attention to improve classification performance.

## 📂 Main File Structure

| File                                  | Description                                                              |
|---------------------------------------|--------------------------------------------------------------------------|
| `Model_CoAtNet.py`                    | CoAtNet model architecture                                               |
| `Model_MobileFormer.py`               | Mobile-Former model architecture                                         |
| `Model_Modification.py`               | Modified model combining Mobile-Former and CoAtNet                       |
| `Flops_Evaluation_[model].py`         | FLOPs-based efficiency evaluation                                        |
| `Train_[model]_on_Imagenet.py`        | Model training on the ImageNet dataset                                   |
| `Train_[model]_on_TinyImagenet.py`    | Model training on the TinyImageNet dataset                               |
| `resize_and_save_to_224.py`           | Image preprocessing for input resizing to 224x224                        |

## 🧠 Training

Model training was conducted using PyTorch. Key training parameters:

- Optimizer: AdamW  
- Learning rate: 1e-3  
- Minimum learning rate: 1e-5  
- Scheduler: Cosine annealing  
- Warmup epochs: 10000  
- Weight decay: 0.05  
- Gradient clipping: 1.0  
- Batch size: 20  
- Total epochs: 300  

Data augmentation techniques applied:
- AutoAugment (rand-m15-n2)  
- Mixup (alpha = 0.8)  
- Label smoothing (0.1)  

## 📈 Evaluation

The main metrics used:

- Top-1 Accuracy  
- FLOPs (Floating Point Operations per Second)

## 📦 Requirements

- PyTorch  
- torchvision  
- timm  
- NVIDIA APEX (optional, for mixed precision training)

## 👥 Team  
**TASI-2425-111**

- 12S21041 – Samuel Christy Angie Sihotang  
- 12S21052 – Griselda  
- 12S21057 – Agnes Theresia Siburian  

Del Institute of Technology  
Information Systems Study Program  
Undergraduate Final Project – Academic Year 2024/2025
