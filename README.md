# Exploring Auto-Annotation with Expert-Crafted Guidelines through 3D LiDAR Detection

![Framework Overview](images/our_pipeline.png) <!-- Replace with your actual image path -->

## Overview

### Research Background & Motivation
- **Current Annotation Paradigm:** Relies on hiring ordinary human annotators to label data based on instructions from expert-crafted guidelines.
- **Inherent Problems:** This paradigm is laborious, tedious, and costly.
- **Research Goal:** To explore **auto-annotation** methods that leverage **expert**-crafted guidelines directly.

### The AutoExpert Benchmark
- **Data Source:** Repurposed from the well-established nuScenes dataset, commonly used in autonomous driving research.
- **Key Features:**
  - Provides authentic, real-world expert-crafted annotation guidelines.
  - Guidelines define 18 object classes using nuanced language descriptions and a few visual examples.
  - Annotation Task: Annotating objects in LiDAR data using 3D cuboids.
  - **Core Challenge:** The guidelines do not provide LiDAR visuals, soliciting methods that learn from few-shot labeled images and texts for 3D detection.

### Technical Challenges
- **Data-Modality Discrepancy:** Bridging the gap between image/text data and 3D point cloud data.
- **Annotation-Task Discrepancy:** Translating 2D detection knowledge to 3D detection tasks.

### Proposed Approach
Leverages publicly available Foundation Models (FMs) in a conceptually simple pipeline:
1.  **2D Object Detection:** Utilize FMs for 2D detection and segmentation in RGB images.
2.  **3D Lifting:** Lift the 2D detections into 3D space using known sensor poses.
3.  **3D Cuboid Generation:** Generate a 3D cuboid for each 2D detection within its corresponding frustum.

### Key Results
- Through progressive refinement of key components, the pipeline achieves a **21.9** 3D detection mAP.
- This performance is significantly higher than the **12.1** mAP achieved by existing methods.
- Demonstrates the promise of Foundation Models for tackling the challenges of automated data annotation.

## Installation
To set up the repository, follow these steps:
1. **Clone the Repository**
    ```bash
    git clone https://github.com/annoguide/annoguide3Dbenchmark.git
    ```
2. **Create the Environment**
   ```bash
    conda create -n annoguide python=3.8
    conda activate annoguide
    ```
3. ​**Install MMDetection**
    ```bash
    conda install pytorch torchvision -c pytorch
    pip install -U openmim
    mim install mmengine
    mim install "mmcv>=2.0.0"
    cd mmdetection
    pip install -v -e .
    ```
4. **Install SAM**
    ```bash
    pip install segment_anything
    ```
5. ​**Install Nuscenes-LT3d**
    ```bash
    git clone https://github.com/neeharperi/nuscenes-lt3d.git
    cd nuscenes-lt3d/setup
    python setup.py install 
    ```
## Data Preparation
1. **NuScenes Dataset**

    Download from [NuScenes](https://www.nuscenes.org/nuscenes)

2. **Few-shot Data from Guidelines**

    Download through [GoogleDrive](https://drive.google.com/file/d/1H0igDf6j0DPHh5YDMKfJ3ZpDbpD8UagC/view?usp=sharing)


3. **Small Validation Set for Prompt/Model Selection**

    Download through [GoogleDrive](https://drive.google.com/file/d/1vHd6W0moZmf0gqQgV-_fHameyGNfqngP/view?usp=sharing)

## Models
1. **GroundingDINO**

    The original model can be downloaded through MMDetection: [MMDetection Repository](https://github.com/open-mmlab/mmdetection)
    
    The finetuned model with refined class name can be downloaded through [GoogleDrive](https://drive.google.com/file/d/1Y0Xcr6u9F8-FmQqR65oQBhMRkMWoHu4m/view?usp=drive_link)


2. **Segment Anything Model (SAM)** 

    Download through official repository: [SAM Repository](https://github.com/facebookresearch/segment-anything)

## Pipeline Execution
### 2D Detection
1. **​Generate COCO-format 2D Labels to Prepare Data Used in AutoExpert**

     ```bash
    python mmdetection/tools/GD/make_2D_labels.py --info_path data/nuscenes/nuscenes_infos_train.pkl --output_dir_path data/nuscenes/samples/labels_2D_COCO/CAM_ALL_train
    python mmdetection/tools/GD/make_2D_labels.py --info_path data/nuscenes/nuscenes_infos_val.pkl --output_dir_path data/nuscenes/samples/labels_2D_COCO/CAM_ALL_val
    ```
2. **Create AutoExpert Training Set** 

     ```bash
    python mmdetection/tools/GD/make_2D_annos_train_few_shot.py
    python mmdetection/tools/GD/make_few-shot_file_name.py
    ```
3. **Create AutoExpert Validation Set** 

     ```bash
    python mmdetection/tools/GD/make_2D_annos_val.py
    ```
4. **Finetune GroundingDINO with Optimized Prompts**

     ```bash
    python mmdetection/tools/train.py mmdetection/configs/mm_grounding_dino/grounding_dino_swin-l_finetune_8xb4_20e_nuscenes_train.py
    ```
5. **Save 2D Detections by the Finetuned GroundingDINO (with Evaluation)**

     ```bash
    python mmdetection/tools/test.py mmdetection/configs/mm_grounding_dino/grounding_dino_swin-l_finetune_8xb4_20e_nuscenes_test.py outputs/nuscenes/weights/epoch_6.pth
    ```
6. **Evaluate the Saved 2D Detections**

     ```bash
    python mmdetection/mmdet/evaluation/metrics/coco_metric.py
    ```

### 3D Cuboids Generation
1. **Generate 2D Masks with SAM** 

     ```bash
    python mmdetection/tools/GD/add_file_name_in_2D_results.py 
    python src/gen_masks.py
    ```
2. **Generate 3D Cuboids for 2D Masks** 

     ```bash
    python src/maks_to_3D_results.py
    ```
3. **Confidence Smoothing via 3D Tracking** 

     ```bash
    python src/post_process/json2pkl.py
    python src/post_process/average_score_by_tracking.py
    ```
4. **3D Evaluation** 

     ```bash
    python src/eval/eval_3D_results.py
    ```    
## Saved Results for Fast Evaluation
1. **2D Detection Results**

    Download through [GoogleDrive](https://drive.google.com/file/d/1p39B3_ZCE5OKRYLZKMifvinYY0kzvqmo/view?usp=sharing)

2. **3D Detection Results** 

    Download through [GoogleDrive](https://drive.google.com/file/d/1aVtFWD6F73CD_dsM3nW2IAVSCKfsAaSz/view?usp=drive_link)
