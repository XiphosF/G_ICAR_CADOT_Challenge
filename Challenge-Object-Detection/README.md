# Hardware & Software requirements

- GPU: PNY NVIDIA RTX A6000 48 GB VRAM (Training can be done on GPUs with less memory by reducing the batch size)
- CUDA: 12.6
- OS: Linux
- Python: 3.8.17

# Dependency installation

Installing MMCV can be tricky due to compatibility issues between CUDA, Python, and PyTorch versions.

1. Create and activate a new conda environment with correct python version.
2. Install dependencies using: `pip install -r requirements.txt`
3. Clone MMDetection v3.1.0 from source and install it in editable mode: `pip install -v -e .` Run this command from inside the mmdetection-3.1.0 directory


The MMDetection (version 3.1.0) is used from source and can finally be installed with  in the mmdetection-3.1.0 folder.

# Step-by-step execution



# Validation score
After fine-tuning, the Faster R-CNN model achieved:
- **mAP@50**: 62.5
- **mAP@50:95**: 38.1



