# 🚗📡 Exploring Auto-Annotation with Expert-Crafted Guidelines through 3D LiDAR Detection

![Framework Overview](images/our_pipeline.png) <!-- Replace with your actual image path -->

---

## 🧠 Overview

### 🔍 Research Background & Motivation

| Aspect | Description |
|--------|-------------|
| **Current Annotation Paradigm** | Relying on hiring **ordinary human annotators** to label data based on **expert-crafted guidelines**. |
| **Inherent Problems** | Labor-intensive, tedious, and **costly**. |
| **Our Research Goal** | To explore **auto-annotation** methods that directly leverage **expert-crafted guidelines**. |

> 💡 We aim to reduce manual effort while preserving annotation quality by enabling machines to understand and apply expert knowledge.

---

### 🏆 The AutoExpert Benchmark

| Feature | Details |
|--------|---------|
| **Data Source** | Repurposed from the well-established [nuScenes dataset](https://www.nuscenes.org), widely used in autonomous driving. |
| **Guidelines** | Authentic, real-world **expert-crafted** annotation rules. |
| **Object Classes** | Defines **18 object classes** using **nuanced textual descriptions** + a few **visual examples**. |
| **Annotation Task** | Annotate 3D objects using **3D cuboids** from LiDAR data. |
| **Core Challenge** | Guidelines contain **no LiDAR visuals**, requiring models to learn from **few-shot labeled images & texts** for 3D detection. |

---

### ⚠️ Technical Challenges

- **🔄 Data-Modality Discrepancy**: Bridging the gap between **2D image/text data** and **3D point clouds**.
- **🎯 Annotation-Task Discrepancy**: Adapting **2D detection knowledge** to complex **3D detection tasks**.

---

### ✅ Proposed Approach

We propose a **simple yet effective pipeline** leveraging **publicly available Foundation Models (FMs)**:

1. **🔍 2D Object Detection**  
   Use FMs for **2D detection & segmentation** on RGB images.

2. **🎯 3D Lifting**  
   Project 2D detections into **3D space** using known **sensor poses**.

3. **📦 3D Cuboid Generation**  
   Generate a **3D bounding box (cuboid)** for each 2D detection within its **camera frustum**.

---

### 📈 Key Results

| Metric | Our Method | Prior Methods |
|--------|------------|---------------|
| **3D Detection mAP** | **21.9** 🚀 | **12.1** |
  
> 🎉 Demonstrates the **great potential** of Foundation Models in automating data annotation with expert guidance!

---

## 🛠️ Installation

Follow the steps below to set up the environment:

1. **📥 Clone the Repository**
    ```bash
    git clone https://github.com/annoguide/annoguide3Dbenchmark.git
    ```
2. **🌱 Create the Conda Environment**
   ```bash
    conda create -n annoguide python=3.8
    conda activate annoguide
    ```
3. **⚙️ Install MMDetection**
    ```bash
    conda install pytorch torchvision -c pytorch
    pip install -U openmim
    mim install mmengine
    mim install "mmcv>=2.0.0"
    cd mmdetection
    pip install -v -e .
    ```
4. **🔒 Install SAM (Segment Anything Model)**
    ```bash
    pip install segment_anything
    ```
5. **📦 Install Nuscenes-LT3d**
    ```bash
    git clone https://github.com/neeharperi/nuscenes-lt3d.git
    cd nuscenes-lt3d/setup
    python setup.py install 
    ```
## 📂 Data Preparation

| Dataset | Download Link |
|--------|----------------|
| **NuScenes Dataset** | [🔗 NuScenes Official](https://www.nuscenes.org/nuscenes) |
| **Few-shot Data from Guidelines** | [🔗 Google Drive](https://drive.google.com/file/d/1H0igDf6j0DPHh5YDMKfJ3ZpDbpD8UagC/view?usp=sharing) |
| **Small Validation Set (for Prompt/Model Selection)** | [🔗 Google Drive](https://drive.google.com/file/d/1vHd6W0moZmf0gqQgV-_fHameyGNfqngP/view?usp=sharing) |

---

## 🤖 Models

| Model | Source / Download |
|-------|-------------------|
| **GroundingDINO** | [📘 MMDetection Repo](https://github.com/open-mmlab/mmdetection) <br> [🔗 Finetuned Model (Google Drive)](https://drive.google.com/file/d/1Y0Xcr6u9F8-FmQqR65oQBhMRkMWoHu4m/view?usp=drive_link) |
| **Segment Anything Model (SAM)** | [🔗 Official Repo](https://github.com/facebookresearch/segment-anything) |

---

## 🔄 Pipeline Execution

### 🖼️ 2D Detection
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

### 🧊 3D Cuboids Generation
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
## 🎮 Demo: Single Sequence Inference & Visualization

### 📊 Demo Notebook: `demo-annoexpert-infer-vis.ipynb`

This Jupyter notebook demonstrates our complete pipeline by processing **1 sample sequence** and visualizing the algorithm workflow step by step.

### 🔄 Pipeline Steps:

#### **Cell 0: Save 2D Detections by the Finetuned GroundingDINO (with Evaluation)**

| Component | Description | Visualization |
|-----------|-------------|---------------|
| **Model Loading** | Load the finetuned GroundingDINO model with optimized prompts | 📥 |
| **2D Detection** | Perform object detection on RGB images by refined class name | 🖼️ + 📦 |
| **Result Saving** | Save 2D bounding boxes with class labels and confidence scores | 💾 |
| **Evaluation** | Calculate 2D detection metrics (mAP, Recall, Precision) | 📊 |

**Output**: 2D bounding boxes overlaid on camera images with confidence scores and categories

#### **Cell 1: Generate 2D Masks with SAM**

| Component | Description | Visualization |
|-----------|-------------|---------------|
| **Mask Generation** | Use Segment Anything Model (SAM) to generate precise segmentation masks | 🎭 |
| **Instance Segmentation** | Convert bounding boxes to detailed instance masks | ✂️ |
| **Mask Refinement** | Refine masks based on image boundaries and object contours | 🎯 |
| **Multi-Object Handling** | Process multiple objects in the same scene | 🔄 |

**Output**: High-quality segmentation masks for each detected object

#### **Cell 2: Generate 3D Cuboids for 2D Masks**

| Component | Description | Visualization |
|-----------|-------------|---------------|
| **3D Lifting** | Project LiDAR points into 2D images | 📐 |
| **Points Estimation** | Estimate LiDAR points for each 2D instance mask | 📏 |
| **Cuboid Generation** | Generate 3D bounding boxes within MHT | 📦 |
| **Multi-View Fusion** | Combine 3D detections from multiple camera views | 🔄 |

**Output**: 3D cuboids in LiDAR coordinate system

#### **Cell 3: Visualize the 2D and 3D results in image and BEV**

| Component | Description | Visualization |
|-----------|-------------|---------------|
| **Image Space Visualization** | Overlay 3D boxes on multiple camera views | 🖼️ |
| **BEV (Bird's Eye View)** | Display 3D cuboids in top-down LiDAR view | 🗺️ |
| **Result Analysis** | Compare auto-annotations with ground truth | 📊 |

**Output**: Comprehensive multi-view visualizations:
- **Camera view**: 3D detections
- **BEV view**: 3D cuboids in point cloud
- **Side-by-side comparisons**

### 🎯 Demo Features:

| Feature | Benefit |
|---------|---------|
| **🔬 Step-by-Step Visualization** | Understand each stage of the auto-annotation pipeline |
| **📊 Intermediate Results** | See how 2D detections transform into 3D annotations |
| **🌍 Multi-Modal Display** | Camera images + LiDAR point clouds + BEV maps |
| **📈 Quality Metrics** | Quantitative assessment of annotation quality |

### 🚀 Quick Start:
```bash
# Launch the demo notebook
jupyter notebook demo-annoexpert-infer-vis.ipynb
```   

## 📦 Saved Results for Fast Evaluation

| Type | Download Link |
|------|---------------|
| **2D Detection Results** | [🔗 Google Drive](https://drive.google.com/file/d/1p39B3_ZCE5OKRYLZKMifvinYY0kzvqmo/view?usp=sharing) |
| **3D Detection Results** | [🔗 Google Drive](https://drive.google.com/file/d/1aVtFWD6F73CD_dsM3nW2IAVSCKfsAaSz/view?usp=drive_link) |

---

✅ **Happy Auto-Annotation with Expert Knowledge！**
