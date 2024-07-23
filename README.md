# Object Detection and Tracking Repository

## Overview
This repository contains the implementation of various object detection and tracking tasks. The activities are divided into two main tasks: Object Detection and Object Tracking. Each task includes multiple sub-tasks to evaluate different models, perform data annotation, fine-tune models, and apply cross-validation techniques.

## Task 1: Object Detection

### Task 1.1: Off-the-shelf - Model Comparison
In this task, we evaluate and compare the performance of three popular object detection models: DETR, Faster R-CNN, and YOLOv8. The comparison includes:

- **Model Setup**: Installation and configuration of each model.
- **Performance Evaluation**: Testing the models on a common dataset.
- **Results Comparison**: Analyzing and comparing the accuracy, speed, and other relevant metrics.


**Pretrained models used** :
DETR (Resnet50 backbone): CNN and transformer based model (implementation).
YoloV8 (YoloV8x version): one-stage CNN based model (implementation).
Faster RCNN (Resnet50_FPNv2 backbone): two-stage CNN based model (implementation).

**Implementation details**:
Executed on a RTX 3060.
All of the models were pre-trained on the COCO dataset. 
We selected both parked and non-parked cars and bicycles. 
Only detected bounding boxes with a confidence score greater than 0.5 were used.


**Conclusions**:
Although YoloV8 had the fastest inference time thanks to being a one-stage CNN, it yielded the worst results. 
Faster RCNN had worse results than DETR, with even longer inference time. 
DETR showed the best performance in terms of mAP and mIoU.



### Task 1.2: Annotation
For this task, we use the Roboflow software to perform data annotation and labeling. This includes:

- **Dataset Preparation**: Gathering and organizing the dataset.
- **Annotation Process**: Using Roboflow to annotate and label the data accurately.
- **Exporting Annotations**: Preparing the labeled data for model training.

### Task 1.3: Fine-tune to Your Data
This task involves fine-tuning the pre-trained models using the annotated dataset from Task 1.2. The steps include:

- **Model Fine-Tuning**: Adjusting the models to better fit the specific dataset.
- **Performance Comparison**: Comparing the fine-tuned models against the pre-trained versions.
- **Analysis**: Evaluating the improvements and identifying any shortcomings.

### Task 1.4: K-Fold Cross-validation
In this task, we apply K-Fold Cross-validation to the dataset using the sklearn implementation. The steps include:

- **Dataset Splits**: Dividing the dataset into K folds with `shuffle=True` for randomization.
- **Cross-validation Process**: Training and validating the models on each fold.
- **Result Analysis**: Aggregating the results and analyzing the model performance across different folds.

## Task 2: Object Tracking
This task involves implementing object tracking techniques to track detected objects across frames in a video sequence. The steps include:

- **Tracker Setup**: Selecting and configuring object tracking algorithms.
- **Tracking Implementation**: Applying the tracking algorithms to video data.
- **Performance Evaluation**: Assessing the accuracy and robustness of the tracking algorithms.

## Getting Started
To get started with this repository, follow the instructions below:

### Prerequisites
- Python 3.x
- Required libraries: [list of libraries]

### Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/object-detection-tracking.git
