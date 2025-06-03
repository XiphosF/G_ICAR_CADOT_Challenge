# Method Overview

This repository highlights the importance of the pretraining phase for object detection models, especially when applied to aerial imagery.

# Baseline Approach

The official baseline provided by the challenge consists of standard object detectors fine-tuned on the target dataset using COCO-pretrained weights. However, COCO is a very different domain compared to aerial imagery, which limits the effectiveness of this pretraining in our context.

# Improved Pretraining Strategy

To obtain a more relevant pretraining starting point, we use the model introduced in the paper:

> "MTP: Advancing Remote Sensing Foundation Model via Multi-Task Pretraining"  
> Di Wang, Jing Zhang, Minqiang Xu, Lin Liu, Dongsheng Wang, Erzhong Gao, Chengxi Han, Haonan Guo, Bo Du, Dacheng Tao and Liangpei Zhang  
> *JSTARS'24*


Their GitHub repository provides ViT/CNN encoders pretrained on a large combination of aerial remote sensing datasets (e.g., DIOR, DOTA, xView, etc.) using a multi-task learning paradigm. These encoders have demonstrated strong generalization across a variety of downstream tasks such as:

- Horizontal and rotated bounding box detection
- Semantic segmentation
- Change detection

# Fine-Tuning Setup

We used the pretrained encoder from MTP and added a Faster R-CNN head for detection. Fine-tuning was performed on the CADOT dataset with the following simple setup:
- 12 training epochs
- Left-right flip as the only data augmentation


# Results 

Despite the simplicity of our fine-tuning setup, the model achieves strong and competitive results, demonstrating the value of domain-aligned pretraining. The baseline Faster R-CNN provided by the CADOT organizers, which uses more extensive data augmentation, reaches a **mAP@50 of 34.46** on the test set. In contrast, our fine-tuned Faster R-CNN model, initialized with MTP-pretrained encoders, achieves a **mAP@50 of 67.25**, highlighting the significant gains that can be achieved through relevant pretraining.

With access to more GPUs and time, the performance of our model could surely be further enhanced by incorporating additional data augmentation techniques tailored to aerial imagery. This includes vertical flips, rotations, photometric distortions, and class-balancing methods such as Copy-Paste augmentation, particularly targeting underrepresented classes in the dataset. Additionally, applying a 5-fold cross-validation strategy (training five separate detectors and combining their predictions using Weighted Boxes Fusion (WBF)) could enhance overall results, though it would require more computational resources.
