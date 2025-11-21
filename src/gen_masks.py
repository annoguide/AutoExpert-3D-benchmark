import sys
import json
import os
import cv2
import random
import time
import numpy as np
import torch
import torchvision
import argparse
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import mini_train, mini_val, train_detect, train, val
from PIL import Image, ImageDraw
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, build_sam
from tqdm import tqdm
import pycocotools.mask
import pickle
from cfg.prompt_cfg import PROMPT_MAPS  
 
import warnings
warnings.filterwarnings("ignore")    

def draw_mask(mask, draw, random_color=False):
    if random_color:
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
            153,
        )
    else:
        color = (30, 144, 255, 153)

    nonzero_coords = np.transpose(np.nonzero(mask))
    for coord in nonzero_coords:
        draw.point(coord[::-1], fill=color)

# utils
def count_frames(nusc, sample):
    frame_count = 1

    if sample["next"] != "":
        frame_count += 1

        sample_counter = nusc.get("sample", sample["next"])

        while sample_counter["next"] != "":
            frame_count += 1
            sample_counter = nusc.get("sample", sample_counter["next"])

    return frame_count

def map_class(name):
    if name in PROMPT_MAPS.keys():
        return PROMPT_MAPS[name]
    sys.exit()
    return False

def parse_args():
    parser = argparse.ArgumentParser(description="NuScenes + SAM Mask Generation Script")

    parser.add_argument("--ver_name", type=str, default="v1.0-trainval",
                        help="NuScenes version name, e.g., v1.0-trainval")
    parser.add_argument("--data_path", type=str, default="data/nuscenes/",
                        help="Path to the NuScenes data directory")
    parser.add_argument("--sam_ckpt", type=str, default="weights/sam_vit_h_4b8939.pth",
                        help="Path to the SAM checkpoint file")
    parser.add_argument("--output_dir", type=str, default="outputs/nuscenes/results_mask/nuscenes-gd-sam/",
                        help="Directory to save the output results")
    parser.add_argument("--gd_path", type=str, default="data/nuscenes/outputs/result_2D_val.json",
                        help="Path to the 2D ground truth or detection JSON file")

    args = parser.parse_args()
    

    return args

if __name__ == "__main__":
    args = parse_args()

    print("[INFO] Running with arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    custom_vocabulary = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'emergency_vehicle',
                        'adult', 'child', 'police_officer', 'construction_worker', 'stroller', 'personal_mobility',
                        'pushable_pullable', 'debris', 'traffic_cone', 'barrier']              

    VER_NAME = args.ver_name
    DATA_PATH = args.data_path
    CAM_LIST = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_FRONT_LEFT",
    ]
    SAM_CKPT = args.sam_ckpt
    OUTPUT_DIR = args.output_dir
    GD_PATH = args.gd_path
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    with open(GD_PATH, "r") as f:
        results_GD = json.load(f)
    results_GD_new = dict()
    for result_GD in results_GD:
        file_name = result_GD['file_name']
        x1, y1, w, h = result_GD['bbox']
        x2, y2 = x1 + w, y1 + h    
        if file_name in results_GD_new:
            results_GD_new[file_name]['boxes_filt_list'].append([x1, y1, x2, y2])
            results_GD_new[file_name]['pred_phrases'].append(f"{custom_vocabulary[result_GD['category_id']-1]}({round(result_GD['score'], 2)})")
        else:
            results_GD_new[file_name] = dict()
            results_GD_new[file_name]['boxes_filt_list'] = list()
            results_GD_new[file_name]['pred_phrases'] = list()
            results_GD_new[file_name]['boxes_filt_list'].append([x1, y1, x2, y2])
            results_GD_new[file_name]['pred_phrases'].append(f"{custom_vocabulary[result_GD['category_id']-1]}({round(result_GD['score'], 2)})") 


    start_time = time.time()
    global blip_processor, blip_model, groundingdino_model, sam_predictor, sam_automask_generator, inpaint_pipeline

    nusc = NuScenes(version=VER_NAME, dataroot=DATA_PATH, verbose=True)

    assert SAM_CKPT, "sam_checkpoint is not found!"
    sam = build_sam(checkpoint=SAM_CKPT)
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)
    sam_automask_generator = SamAutomaticMaskGenerator(sam)

    # make output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sample_count = 1

    for scene_num, scene_name in enumerate(val):
        scene_token = nusc.field2token("scene", "name", scene_name)[0]
        scene = nusc.get("scene", scene_token)
        sample = nusc.get("sample", scene["first_sample_token"])

        num_frames = count_frames(nusc, sample)
        for f in tqdm(range(num_frames), desc=scene_name+": "):
            frame_start_time = time.time()
            cam_nums = []
            labels = []
            detection_scores = []
            im_to_ego_mats = []
            mask_images = []
            data = {}
            np_mask_images = []

            for c in range(len(CAM_LIST)):
                this_cam_labels = []
                this_cam_scores = []

                (path, nusc_boxes, camera_instrinsic) = nusc.get_sample_data(
                    sample["data"][CAM_LIST[c]]
                )

                # Use GD 2D results
                detic_start = time.time()
                file_name = path.split('/')[-1]
                if file_name in results_GD_new:
                    image_pil = Image.open(path).convert('RGB')
                    image = np.array(image_pil)
                    image = image[:, :, ::-1].copy()  
                    boxes_filt_list = results_GD_new[file_name]['boxes_filt_list']
                    pred_phrases = results_GD_new[file_name]['pred_phrases']                
                    boxes_filt = torch.tensor(boxes_filt_list).cuda()
                else:
                    print("\nNo box found in this image.\n")
                    im_to_ego_mats.append([])
                    continue
                    
                detic_end = time.time()

                # visualize pred
                size = image_pil.size
                pred_dict = {
                    "boxes": boxes_filt,
                    "size": [size[1], size[0]],  # H, W
                    "labels": pred_phrases,
                }
                np_image = np.array(image_pil)
                boxes_filt = boxes_filt.cpu()

                # implement class-wise 2D nms on these boxes
                nms_start = time.time()
                for i, (box, label) in enumerate(zip(boxes_filt, pred_phrases)):
                    idx = label.find("(")
                    ds = label[idx + 1 : -1]
                    label = label[:idx]

                    this_cam_labels.append(map_class(label.lower()))
                    this_cam_scores.append(float(ds))
            
                """ Unomment for NMS """
                run_nms = True
                if run_nms:
                    nms_scores = []
                    nms_boxes_filt = torch.Tensor([])
                    nms_labels = []

                    for cls in set(PROMPT_MAPS.values()):
                        cls_boxes = boxes_filt[[i for i, l in enumerate(this_cam_labels) if l == cls]]
                        cls_scores = [this_cam_scores[i] for i, l in enumerate(this_cam_labels) if l == cls]

                        if len(cls_boxes) == 0:
                            continue

                        keep = torchvision.ops.nms(cls_boxes, torch.Tensor(cls_scores), 0.75)
                        keep = keep.cpu()
                        cls_boxes = cls_boxes[keep]
                        cls_scores = [cls_scores[i] for i in keep]
                        cls_labels = [cls for _ in range(len(keep))]
                        nms_boxes_filt = torch.cat((nms_boxes_filt, cls_boxes), dim=0)
                        nms_scores.extend(cls_scores)
                        nms_labels.extend(cls_labels)

                    boxes_filt = nms_boxes_filt
                    detection_scores.extend(nms_scores)
                    labels.extend(nms_labels)

                else:
                    detection_scores.extend(this_cam_scores)
                    labels.extend(this_cam_labels)
                    nms_labels = this_cam_labels
                """ """
                
                nms_end = time.time()
                    
                sam_start = time.time()
                sam_predictor.set_image(np_image)
                
                transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(DEVICE)
                if transformed_boxes.shape[0] == 0:
                    print("No objects found.")
                    im_to_ego_mats.append([])
                    continue
                

                masks, _, _ = sam_predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                )

                sam_end = time.time()

                assert len(boxes_filt) == len(nms_labels) == len(masks)

                for i, (box, label, mask) in enumerate(zip(boxes_filt, pred_phrases, masks)):
                    mask_image = Image.new("RGBA", size, color=(0, 0, 0, 0))
                    mask_draw = ImageDraw.Draw(mask_image)
                    draw_mask(mask[0].cpu().numpy(), mask_draw)
                    cam_nums.append(c)

                    np_mask_image = np.array(mask_image).astype(np.uint8).transpose([2, 1, 0])[3, :, :]
                    np_mask_image = np.squeeze(np_mask_image)

                    compressed_np_mask_image = pycocotools.mask.encode(np.asfortranarray(np_mask_image))
                    np_mask_images.append(compressed_np_mask_image)

            np_images = np_mask_images

            if len(labels) == 0:
                continue

            assert len(labels) == len(detection_scores)
            assert len(labels) == len(cam_nums)
            assert len(np_images) == len(labels)

            data["labels"] = labels
            data["detection_scores"] = detection_scores
            data["cam_nums"] = cam_nums
            

            os.makedirs(os.path.join(OUTPUT_DIR, scene_name), exist_ok=True)
            with open(os.path.join(OUTPUT_DIR, f"{scene_name}", f"{f}_data.json"), "w") as outfile:
                json.dump(data, outfile)

            pickle.dump(np_images, open(os.path.join(OUTPUT_DIR, f"{scene_name}", f"{f}_masks.pkl"), 'wb'))

            if sample['next'] != "":
                sample = nusc.get('sample', sample['next'])
                sample_count += 1
            frame_end_time = time.time()
            print(f"Took {frame_end_time - frame_start_time} seconds for a sample.")
    end_time = time.time()
    print(f"Took {end_time - start_time} seconds for {sample_count} samples.")

