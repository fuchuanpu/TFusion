# TFusion

**Generic Feature Extraction for Few-Shot Malicious Traffic Detection.**

This repository is anonymous and contains the code and datasets used in the submitted paper. 
We provide a minimal working demo to reproduce the majority of the results in the paper.

> This repository and associated dataset are anonymous for the double-blind review process.
> We are grateful to the anonymous sharing services `file.io` and `anonymous.4open.science`.

## 0x0 Hardware
The code has been tested on a clean `Ubuntu 22.04` machine. Please ensure you have around 10GB of free disk space available for the dataset and the model.

## 0x1 Installation
### 0x11 Install Environments:
``` bash
bash ./scripts/env.sh
```
The current version supports GPU execution (up to four GPUs), but the installation only supports the CPU version of PyTorch. Please modify it for GPU support.

### 0x12 Download the Datasets 
The dataset is 463MB before extraction and 5.2GB after extraction:
``` bash
TARGET_HASH="bv9vklT88pyS"
wget https://file.io/$TARGET_HASH -O tfusion_data.tar.gz
tar -xvzf tfusion_data.tar.gz
rm $_
```

Please try a different `TARGET_HASH` if the link has expired, due to the one-download expiration policy of anonymous file sharing at `file.io`:
1. Kzd1nNa7SunE
2. TcVGgmupuhoM
3. Kzd1nNa7SunE

## 0x2 Usage
### 0x21 Construct a TFusion model
``` bash
chmod +x ./main.py
./main.py -c ./config/pre-train/MAWI_F.json
```
The pre-train model is saved to `./save/traffic_model/mawi/20230101_F_model.pt`

### 0x22 End-to-End Detection
We use TFusion-extracted features to train lightweight models and test their performance on the test set:
``` bash
chmod +x script/run_all.py
./script/run_all.py
```
The detailed results can be found in `./log` and `./figure`.

## 0x3 Reference
```
Only 1.0‰ Traffic Samples are Required for Malicious Traffic Detection: A Cross-Attention Feature Fusion Based Few-Shot Learning, Anonymous authors, 2024.
```