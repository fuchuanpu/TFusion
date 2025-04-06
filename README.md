# tFusion

**Cross-Modality Feature Fusion for Malicious Traffic Detection.**

This repository is anonymous and contains the code and datasets used in the submitted paper. 
We provide a minimal working demo to reproduce the majority of the results in the paper.

> This repository and associated dataset are anonymous for the double-blind review process.
> We are grateful to the anonymous sharing services `anonymous.4open.science`.

## 0x0 Hardware
The code has been tested on a clean `Ubuntu 22.04` machine. Please ensure you have around 10GB of free disk space available for the dataset and the model.

## 0x1 Installation
### 0x11 Install Environments:
``` bash
bash ./scripts/env.sh
```
The current version supports GPU execution (up to four GPUs), but the installation only supports the CPU version of PyTorch. Please modify it for GPU support.

### 0x12 Download the Datasets 
Please download the datsets from the HotCRP. The dataset is 443MB before extraction and around 5.2GB after extraction:
``` bash
tar -xvzf tfusion_data.tar.gz
rm $_
```

## 0x2 Usage
### 0x21 Construct a tFusion model
``` bash
chmod +x ./main.py
./main.py -c ./config/pre-train/MAWI_F.json
```
The pre-train model is saved to `./save/traffic_model/mawi/20230101_F_model.pt`

### 0x22 End-to-End Detection
We use tFusion-extracted features to train lightweight models and test their performance on the test set:
``` bash
chmod +x script/run_all.py
./script/run_all.py
```
The detailed results can be found in `./log` and `./figure`.

## 0x3 Reference
> "Training with Only 1.0 ‰ Samples: Malicious Traffic Detection via Cross-Modality Feature Fusion", Anonymous authors, 2025.
