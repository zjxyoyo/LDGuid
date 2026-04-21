# LDGuid: A Framework for Robust Change Detection via Latent Difference Guidance

[![Conference](https://img.shields.io/badge/Accepted-IGARSS%202026-blue)](#) 
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 📖 Introduction

This repository contains the official PyTorch implementation of the paper **"LDGUID: A FRAMEWORK FOR ROBUST CHANGE DETECTION VIA LATENT DIFFERENCE GUIDANCE"**.

Modern deep learning models for change detection (CD) often struggle to explicitly represent task-relevant semantic differences. This challenge is particularly evident in applications requiring soft information on scene changes rather than simple binary change maps (e.g., wildfire segmentation).

To address this, we propose the **Latent Difference Guidance (LDGuid)** framework. The main contributions of our work include:
* **Difference Embedding (DE) Module**: We design a DE module utilizing adversarial autoencoding to learn a robust latent representation of semantic differences.
* **Information Bottleneck Pretraining**: The DE module is pretrained via the information bottleneck method, restricting it to learn *only* task-relevant differences between pre- and post-event samples, thereby minimizing background noise.
* **Explicit Guidance Mechanism**: The learned latent difference is explicitly injected into the downstream CD model as a guidance signal.
* **Versatility and Robustness**: LDGuid can be seamlessly integrated into various architectures. We validate it using U-Net, BIT, and AERNet baselines across the LEVIR-CD, WHU-CD, SVCD, and CaBuAr datasets. Experimental results demonstrate consistent segmentation performance enhancements, with remarkable gains in challenging settings affected by heavy spectral noise, such as burn scar detection.
<img width="576" height="328" alt="Figure1" src="https://github.com/user-attachments/assets/14d234f3-bdcf-4e96-ba81-82497955995f" />

### ✨ Visual Results
<img width="1347" height="540" alt="Visual_Final" src="https://github.com/user-attachments/assets/f07a330b-165d-4549-a896-a8b2e75e6f09" />



## ⚙️ Environment Setup

This project was initially developed and tested on the Compute Canada High-Performance Computing (HPC) cluster. For the open-source community, we provide a generalized environment setup guide to ensure seamless reproduction on local machines or personal servers.

### 1. Create a Virtual Environment (Recommended)
We highly recommend using Conda or venv to create an isolated Python environment (Python 3.10+ is recommended):

```bash
# Create and activate a new Conda environment
conda create -n ldguid python=3.10
conda activate ldguid
```

### 2. Install Dependencies
The core dependencies can be installed via the provided requirements.txt
```bash
pip install -r requirements.txt
```

_Note: The requirements.txt includes standard versions of scientific libraries (NumPy, Pandas, etc.). For **PyTorch**, please ensure you install the version compatible with your local CUDA setup by following the instructions at pytorch.org._

## 🗂️ Data Preparation

Please download the original datasets from their official sources:
- [LEVIR-CD](https://justchenhao.github.io/LEVIR/)
- [WHU-CD](http://gpcv.whu.edu.cn/data/building_dataset.html)
- [SVCD](https://drive.google.com/file/d/1GX656JqqOyBi_Ef0w65kDGVto-nHrNs9)
- [CaBuAr](https://huggingface.co/datasets/DarthReca/california_burned_areas)

Due to the distinct data loading mechanisms of the different baselines implemented in this framework, you must manually organize your downloaded datasets into the exact directory structures shown below before running any training scripts.

### 1. Data Structure for AERNet and U-Net
For the **AERNet** and **U-Net** models, the required data structure depends on the specific dataset.

**For the CaBuAr Dataset:**
After extracting the official downloaded file, place the two main folders into the `wildfire-segmentation/data/` directory.
```text
wildfire-segmentation/
└── data/
    └── CaBuAr/
        ├── pre_post_fire/   # Contains pre-fire, post-fire images, and masks (used for training)
        └── post_fire/       # Contains post-fire images and masks only
```

**For SVCD, LEVIR-CD, and WHU-CD Datasets:**
Ensure each dataset folder contains exactly three subfolders for the bitemporal images and their corresponding labels.
```text
[Model_Directory]/           # e.g., AERNet/ or U-Net_Directory/
└── data/
    └── [Dataset_Name]/      # e.g., SVCD/ or LEVIR-CD/
        ├── A/               # Pre-event images
        ├── B/               # Post-event images
        └── label/           # Change masks (Ground truth)
```

**2. Data Structure for BIT**
The **Bitemporal Image Transformer (BIT)** model employs a different data loading logic. **All 4 datasets** (including CaBuAr) must be manually unified into the following structure.

Crucially, you must create a list directory containing .txt files that explicitly define the image filenames for each data split.
```text
BIT_CD/
└── data/
    └── [Dataset_Name]/      # e.g., LEVIR-CD/, CaBuAr/, etc.
        ├── A/               # Pre-event images
        ├── B/               # Post-event images
        ├── label/           # Change masks (Ground truth)
        └── list/            # Text files defining data splits
            ├── train.txt    # List of image names used for training
            ├── val.txt      # List of image names used for validation
            └── test.txt     # List of image names used for testing
```

## 🚀 Quick Start

We provide a step-by-step guide to train the LDGuid framework using the **CaBuAr** dataset with the **U-Net** baseline. Once you are familiar with this main pipeline, you can easily explore other datasets and model combinations.

### 1. Main Pipeline: Wildfire Segmentation (CaBuAr + U-Net)

**Step 1: Pretrain the Difference Embedding (DE) Module**
First, train the adversarial autoencoder to learn the latent representation of semantic differences from the bitemporal images. Navigate to your project directory and run:
```bash
python src/train/modified_autoencoder_train.py
```
_This script will save the pretrained DE model weights to your local directory._
**Step 2: Train the Main CD Model (LDGuid U-Net)**
Next, train the downstream U-Net model guided by the pretrained DE module.
```bash
python src/train/modified_train.py
```
_(⚠️ Important: Before running this script, please open it and ensure that the load path for the DE weights correctly points to the file generated in Step 1)._

**Step 3: Evaluation**
The training script will automatically generate .csv log files containing the evaluation metrics (e.g., IoU, F1-score) across epochs. You can review these CSV files to evaluate the model's performance.

## 2. Advanced Usage: Exploring Other Combinations

For urban building change detection tasks, we provide scripts to run both our proposed **LDGuid** models and the pure baseline models (**without DE**) for fair comparison.

### 🔹 Option A: U-Net + SVCD

**Pretrain DE Module:**
```bash
python train_ae_svcd.py
```

**Train Proposed LDGuid U-Net:**
```bash
python svcd_train_unet_with_latent.py
```

**Train Pure Benchmark (For Comparison):**
```bash
python svcd_train_unet_benchmark.py
```

### 🔹 Option B: BIT + SVCD

**Pretrain DE Module:**
You can reuse the DE weights generated by ```train_ae_svcd.py``` to save computational time.

**Train Proposed LDGuid BIT:**
```bash
python train_latent_bit_SVCD.py
```

**Train Pure Benchmark (For Comparison):**
```bash
python main_cd.py
```
**Note:** When running ```main_cd.py```, ensure that the dataset directory is correctly specified in the script's arguments.




⚠️ **Special Note on Slurm Scripts (.sh files)**

You may notice several ```.sh``` files in this repository. These are Slurm job submission scripts originally used for the Compute Canada cluster.
If you are running the code on a local machine or standard server: Please ignore these .sh files. You can execute the Python scripts directly.
