This repository presents the G_ICAR team's approach for the CADOT Challenge at ICIP 2025.
Please note that the exact file structure required by the challenge guidelines could not be fully followed due to constraints imposed by the OpenMMLab framework.

# Hardware & Software requirements

- GPU: PNY NVIDIA RTX A6000 48 GB VRAM (Training can be done on GPUs with less memory by reducing the batch size)
- CUDA: 12.6
- OS: Linux
- Python: 3.8.17

# Dependency installation

Installing MMCV can be tricky due to compatibility issues between CUDA, Python, and PyTorch versions.

1. Create and activate a new conda environment with correct python version.
2. Install dependencies using: `pip install -r requirements.txt`
3. Install MMDetection v3.1.0 from source in editable mode: `pip install -v -e .` Run this command from inside the mmdetection-3.1.0 directory.



The pretrained encoder weights from MTP, as well as the final Faster R-CNN model fine-tuned on the CADOT dataset, can be downloaded from the following link: https://drive.google.com/drive/folders/1tbRPtCLdhGsWlDx_vaZJQUxmdwtWwtSi?usp=drive_link

# Step-by-step execution

First, place the pretrained MTP encoder weights (`last_vit_l_rvsa_ss_is_rd_pretrn_model.pth`) in the MTP folder.
Optionally, to run inference directly, place the fine-tuned Faster R-CNN weights (`epoch_12.pth`) in "mmdetection-3.1.0/work_dir_fasterrcnn_finetuned" directory.

Once the environment is properly set up, you can begin fine-tuning the Faster R-CNN model using the MTP pretrained weights with the following steps:

1. Modify your path to the dataset and pretrained weights in the config file of the model "RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/icipchallenge/faster_rcnn_rvsa_l_800_mae_mtp_icip.py"
   
2. Navigate to the "MTP/mmdetection-3.1.0" directory and run:
`python tools/train.py ../RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/icipchallenge/faster_rcnn_rvsa_l_800_mae_mtp_icip.py --work-dir work_dir_fasterrcnn_finetuned`

3. After fine-tuning, run inference on the test images: 
`python demo/image_demo.py /path/to/the/test/images work_dir_fasterrcnn_finetuned/DATE_REPLACE/vis_data/config.py --weights work_dir_fasterrcnn_finetuned/epoch_12.pth --out-dir ./results_pred` . Don't forget to replace the "DATE_REPLACE" by the one output by the training.

4. Finally, generate the submission JSON file for the challenge with: 
`python create_submission.py`


# Validation score
After fine-tuning, the Faster R-CNN model achieved:
- **mAP@50**: 62.5
- **mAP@50:95**: 38.1
