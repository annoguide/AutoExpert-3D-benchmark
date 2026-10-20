<h1 align="center">
  AutoExpert: Automating 3D LiDAR Annotation<br>
  from Expert-Crafted Guidelines
</h1>

<p align="center">
  <strong>Accepted by NeurIPS 2026</strong>
</p>

<p align="center">
  <strong>Yechi Ma</strong><sup>1</sup>&nbsp;&nbsp;
  <strong>Wei Hua</strong><sup>1</sup>&nbsp;&nbsp;
  <strong>Shu Kong</strong><sup>2,3</sup>
</p>

<p align="center">
  <sup>1</sup>Zhejiang University&nbsp;&nbsp;
  <sup>2</sup>University of Macau&nbsp;&nbsp;
  <sup>3</sup>IAIBS
</p>

<p align="center">
  <a href="https://arxiv.org/pdf/2506.02914v3">Paper</a>&nbsp;·&nbsp;
  <a href="https://annoguide.github.io/AutoExpert-3D-benchmark/">Project Page</a>&nbsp;·&nbsp;
  <a href="https://huggingface.co/spaces/autoexpert-cvpr2026-workshop/auto3D">Challenge &amp; Leaderboard</a>&nbsp;·&nbsp;
  <a href="https://huggingface.co/datasets/autoexpert-cvpr2026-workshop/dataset-public">Dataset</a>
</p>

---

AutoExpert studies **auto-annotation from expert-crafted guidelines**: learning from textual class definitions and a few labeled RGB examples to annotate LiDAR data with 3D cuboids. The paper introduces two benchmarks, **AutoExpert-nuScenes** and **AutoExpert-PandaSet**, and the **auto3D** method.

This repository was originally released under the anonymous account [annoguide](https://github.com/annoguide/AutoExpert-3D-benchmark). The personal repository is [cc50121/AutoExpert-3D-benchmark](https://github.com/cc50121/AutoExpert-3D-benchmark).

![AutoExpert pipeline](images/our_pipeline.png)

## Benchmarks and protocol

| | AutoExpert-nuScenes | AutoExpert-PandaSet |
| :--- | :--- | :--- |
| Source dataset | [nuScenes](https://www.nuscenes.org/nuscenes) | [PandaSet](https://github.com/scaleapi/pandaset-devkit) |
| Target categories | **18** guideline-defined classes | **25** guideline-defined classes |
| Training supervision | Guideline texts and few-shot, federated 2D annotations | Guideline texts and few-shot, federated 2D annotations |
| Validation | **570** frames sampled from the official nuScenes training split | **8** re-annotated keyframes |
| Test | **6,019** frames from the official nuScenes validation split | **192** re-annotated keyframes, following the split in paper §3 |
| Annotation basis | nuScenes annotations with the AutoExpert evaluation protocol | **Our corrected and re-annotated PandaSet subset**, comprising **200 non-continuous keyframes and 4,695 cuboids** in total |
| Paper results | Main paper, Table 1 | Appendix L, Table 22 |
| Workshop challenge | Development benchmark | **Official CVPR 2026 Workshop Challenge benchmark** |

The training examples use **4–8 selected RGB images per class** from the source training sets to represent guideline examples. Their annotations are **federated**: only the target class is labeled in each example image. The benchmark does not use the potentially copyrighted example images embedded in the original guidelines.

**No 3D annotations are used for training.** Validation annotations support prompt/model selection and expert-in-the-loop quality control; they are separate from the held-out test annotations. AutoExpert evaluates all 18 nuScenes guideline classes rather than the standard 10-class nuScenes detection task.

### What was corrected in PandaSet?

AutoExpert-PandaSet uses a manually re-annotated subset rather than the original PandaSet cuboids unchanged. As described in Appendix L and Figure 12, the original annotations contain misaligned or "ghost" cuboids whose objects are not identifiable in the RGB views. We re-annotated the selected frames with well-aligned 3D boxes for visually identifiable objects. The [challenge dataset documentation](https://huggingface.co/datasets/autoexpert-cvpr2026-workshop/dataset-public) describes the released 2D examples, validation annotations, and test inputs.

### Evaluation metrics

- **3D mAP** averages per-class AP over ground-plane center-distance thresholds of **0.5, 1.0, 2.0, and 4.0 meters**.
- **nuScenes NDS** combines mAP with translation, scale, orientation, velocity, and attribute errors.
- **PandaSet NDS** is adapted to use translation, scale, and orientation errors, **excluding velocity and attribute errors** (paper Appendix L, Eq. 6). NDS values from the two datasets should be interpreted within their respective protocols.
- **2D mAP** is measured at **IoU = 0.5**.

All mAP and NDS values in the tables below are shown on a **0–100 scale**; the challenge server displays scores on a 0–1 scale.

## Method: auto3D

The paper organizes the pipeline into three components:

1. **2D detection and segmentation.** Refine class names with a VLM using guideline texts and examples, select prompts on the validation set, and finetune GroundingDINO with the few-shot 2D examples. SAM produces instance masks.
2. **2D-to-3D association.** Use known camera/LiDAR calibration and sensor poses to associate the 2D detections with LiDAR points.
3. **3D cuboid generation and refinement.** Use VLM-guided Multiple Hypotheses Testing (**v-MHT**) with instance-specific geometric priors, class-aware LiDAR sweep aggregation, geometry-aided scoring, and tracking-based confidence refinement.

**auto3D is the complete method; v-MHT is its cuboid-generation component.** On AutoExpert-nuScenes, v-MHT alone reaches **21.9 mAP / 25.2 NDS**, while the complete auto3D system reaches **25.4 mAP / 27.2 NDS** (paper Table 4).

## Results reported in the paper

### AutoExpert-nuScenes — 18 classes (Table 1)

| Method | 3D mAP ↑ | NDS ↑ |
| :--- | ---: | ---: |
| Oyster | 6.3 | 10.7 |
| LISO | 8.9 | 13.1 |
| UNION | 9.7 | 13.8 |
| CPD | 10.1 | 14.2 |
| Find&Prop | 11.3 | 16.0 |
| OpenSight | 11.8 | 16.4 |
| OpenBox | 10.6 | 15.5 |
| AnnofreeOD | 11.7 | 16.2 |
| CM3D | 12.1 | 16.6 |
| CM3D + ft-GD | 18.2 | 23.1 |
| **auto3D (ours)** | **25.4** | **27.2** |

Here, **ft-GD** denotes our GroundingDINO finetuned with refined class names and few-shot guideline examples. Table 1 in the paper also reports enhanced variants of the other baselines.

### AutoExpert-PandaSet — our corrected annotations, 25 classes (Table 22)

These results use **AutoExpert-PandaSet**, not the original PandaSet annotation protocol.

| Method | 3D mAP ↑ | Adapted NDS ↑ |
| :--- | ---: | ---: |
| OpenBox | 10.9 | 15.0 |
| OpenBox + ft-GD | 15.9 | 18.9 |
| AnnofreeOD | 11.5 | 15.4 |
| AnnofreeOD + ft-GD | 16.1 | 19.3 |
| CM3D | 12.3 | 16.0 |
| CM3D + ft-GD | 16.6 | 19.7 |
| **auto3D (ours)** | **18.4** | **27.6** |

## CVPR 2026 Workshop Challenge

The [Automated 3D Cuboid Annotation on LiDAR Data challenge](https://huggingface.co/spaces/autoexpert-cvpr2026-workshop/auto3D) evaluates submissions on **AutoExpert-PandaSet with our corrected annotations**. The leaderboard ranks submissions by **NDS**.

The following standings were retrieved from the challenge leaderboard on **October 20, 2026**. The public and private leaderboard views returned the same standings at that time.

| Leaderboard rank | Team / entry | NDS ↑ | 3D mAP ↑ |
| ---: | :--- | ---: | ---: |
| **1** | **iMerit** | **29.36** | **23.24** |
| 2 | AnnoExpert (organizer baseline) | 27.61 | 18.36 |
| 3 | EgoCar | 23.17 | 18.13 |
| 4 | Weaver | 21.03 | 18.07 |
| 5 | LogicCrafters | 20.89 | 17.68 |

**iMerit is the first-place entry on the linked leaderboard.** The organizer baseline is listed under the name **AnnoExpert**; its 18.36 mAP / 27.61 NDS round to the paper's PandaSet auto3D result of 18.4 / 27.6. Challenge submissions are listed separately from the paper's baseline comparisons. Visit the live leaderboard for the complete standings.

## Data and released artifacts

### AutoExpert-nuScenes

| Resource | Link |
| :--- | :--- |
| Source sensor data | [nuScenes official download](https://www.nuscenes.org/nuscenes) |
| Few-shot guideline examples | [Google Drive](https://drive.google.com/file/d/1H0igDf6j0DPHh5YDMKfJ3ZpDbpD8UagC/view?usp=drive_link) |
| Small validation set for prompt/model selection | [Google Drive](https://drive.google.com/file/d/1vHd6W0moZmf0gqQgV-_fHameyGNfqngP/view?usp=drive_link) |
| Finetuned GroundingDINO weights | [Google Drive](https://drive.google.com/file/d/1Y0Xcr6u9F8-FmQqR65oQBhMRkMWoHu4m/view?usp=drive_link) |
| Saved 2D detection results (694.6 MB) | [Google Drive](https://drive.google.com/file/d/1p39B3_ZCE5OKRYLZKMifvinYY0kzvqmo/view?usp=drive_link) |
| Saved 3D detection results (396.1 MB) | [Google Drive](https://drive.google.com/file/d/1aqK5FFMZhofyxYRVvQ1TfTUuz2YhHOAq/view?usp=drive_link) |

The existing Google Drive artifacts belong to the **nuScenes workflow**. They are not PandaSet challenge predictions. Their availability does not by itself verify reproduction of the final paper's auto3D scores.

### AutoExpert-PandaSet

| Resource | Link |
| :--- | :--- |
| Guideline examples, validation annotations, and test inputs | [Hugging Face dataset](https://huggingface.co/datasets/autoexpert-cvpr2026-workshop/dataset-public) |
| LiDAR sequences and sensor data | [Hugging Face sequences](https://huggingface.co/datasets/autoexpert-cvpr2026-workshop/seq) |
| PandaSet loader and calibration utilities | [PandaSet devkit](https://github.com/scaleapi/pandaset-devkit) |
| Submission format, evaluation server, and leaderboard | [Challenge Space](https://huggingface.co/spaces/autoexpert-cvpr2026-workshop/auto3D) |

Consult the challenge dataset card for the current annotation release status and exact submission format. The source datasets remain subject to their original license terms.

## Installation and code availability

Clone the personal repository:

```bash
git clone https://github.com/cc50121/AutoExpert-3D-benchmark.git
cd AutoExpert-3D-benchmark
conda create -n autoexpert python=3.10
conda activate autoexpert
```

Install a compatible [PyTorch/CUDA build](https://pytorch.org/get-started/locally/) and follow the [bundled MMDetection installation guide](mmdetection/docs/en/get_started.md). Use compatible MMEngine/MMCV versions for the bundled MMDetection checkout.

```bash
pip install -U openmim
mim install "mmengine>=0.7.1,<1.0.0"
mim install "mmcv>=2.0.0,<2.2.0"
cd mmdetection
pip install -v -e .
cd ..
pip install segment_anything numba
```

Models used in the paper include [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [SAM](https://github.com/facebookresearch/segment-anything), GPT-4o, and Qwen. Local VLM serving can use [vLLM](https://github.com/vllm-project/vllm); configure it in an environment compatible with the chosen model and serving version.

**Current code coverage.** The released scripts primarily implement the nuScenes workflow. The current tree includes class-level shape priors in [shape_priors_gpt4.json](src/cfg/shape_priors_gpt4.json), but does not include the previously documented `src/vlm_geometric_reasoning.py` entry point. The commands below describe the available scripts; they do not establish that the complete final v-MHT/auto3D pipeline has been released or reproduced. For PandaSet, use the challenge resources above.

## Available nuScenes workflow

Run commands from the repository root. Before running, review paths, checkpoints, and output locations in the scripts and configs. The workflow expects prepared nuScenes info files, downloaded model weights, and SAM checkpoints. The evaluation script expects the extended **18-class** nuScenes evaluation setup (`detection_lt3d`), which is not the standard 10-class devkit configuration.

### 1. Prepare the few-shot and validation data

Generate 2D labels used by the data preparation scripts:

```bash
python mmdetection/tools/GD/make_2D_labels.py --info_path data/nuscenes/nuscenes_infos_train.pkl --output_dir_path data/nuscenes/samples/labels_2D_COCO/CAM_ALL_train
python mmdetection/tools/GD/make_2D_labels.py --info_path data/nuscenes/nuscenes_infos_val.pkl --output_dir_path data/nuscenes/samples/labels_2D_COCO/CAM_ALL_val
python mmdetection/tools/GD/make_2D_annos_train_few_shot.py
python mmdetection/tools/GD/make_few_shot_file_name.py
python mmdetection/tools/GD/make_2D_annos_val.py
```

These preparation utilities use existing source annotations to construct the prescribed examples/evaluation files. Training must follow the benchmark's few-shot 2D supervision protocol; do not use dense source 3D labels as training supervision.

### 2. Finetune and evaluate GroundingDINO

```bash
python mmdetection/tools/train.py mmdetection/configs/mm_grounding_dino/grounding_dino_swin-l_finetune_8xb4_20e_nuscenes_train.py
python mmdetection/tools/test.py mmdetection/configs/mm_grounding_dino/grounding_dino_swin-l_finetune_8xb4_20e_nuscenes_test.py weights/GD.pth
```

### 3. Generate masks and 3D cuboids with the released implementation

```bash
python mmdetection/tools/GD/add_file_name_in_2D_results.py
python src/gen_masks.py
python src/masks_to_3D_results.py
```

### 4. Refine confidence scores and evaluate

```bash
python src/post_process/json2pkl.py
python src/post_process/average_score_by_tracking.py
python src/eval/eval_3D_results.py
```

See [demo-annoexpert-infer-vis.ipynb](demo-annoexpert-infer-vis.ipynb) for the released inference/visualization demo.

## Citation

Please use the updated paper title:

```bibtex
@inproceedings{ma2026autoexpert,
  title     = {{AutoExpert}: Automating 3D {LiDAR} Annotation from Expert-Crafted Guidelines},
  author    = {Ma, Yechi and Hua, Wei and Kong, Shu},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  url       = {https://arxiv.org/pdf/2506.02914v3}
}
```

## License

Repository code is provided under the [MIT License](LICENSE). Please also comply with the applicable source dataset and model licenses.
