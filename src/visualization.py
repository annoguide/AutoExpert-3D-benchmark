import sys
import cv2
import os
import argparse
import tqdm
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box as NuScenesBox
from nuscenes.utils.splits import val
from nuscenes.utils.color_map import get_colormap
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from pyquaternion.quaternion import Quaternion
import pickle
import json
import pycocotools
from PIL import Image
import numpy as np

# Define constants
VIEWS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
         'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

CUSTOM_VOCABULARY = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'emergency_vehicle',
                    'adult', 'child', 'police_officer', 'construction_worker', 'stroller', 'personal_mobility',
                    'pushable_pullable', 'debris', 'traffic_cone', 'barrier']

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Visualize 3D detection results')
    parser.add_argument('--sample-token', type=str, default=None, 
                       help='Specific sample token to visualize. If None, process all samples')
    parser.add_argument('--score-threshold', type=float, default=0.3,
                       help='Detection score threshold. Results below this threshold will be filtered')
    parser.add_argument('--save-interval', type=int, default=10,
                       help='Save interval. Save results every N samples')
    parser.add_argument('--views', nargs='+', default=['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                       'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
                       help='List of camera views to process')
    parser.add_argument('--save-dir', type=str, default='outputs/nuscenes/results_3D/visualiztions/',
                       help='Directory to save results')
    parser.add_argument('--gd-path', type=str, default="data/outputs/result_2D_val.json",
                       help='Path to 2D detection results file')
    parser.add_argument('--res3d-path', type=str, default='outputs/nuscenes/results_3D/result_3D_val.json',
                       help='Path to 3D detection results file')
    parser.add_argument('--nuscenes-root', type=str, default='data/nuscenes/',
                       help='NuScenes dataset root directory')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                       help='NuScenes dataset version')
    parser.add_argument('--use-gd', action='store_true', default=True,
                       help='Whether to use 2D detection results')
    
    return parser.parse_args()

def count_frames(nusc, sample):
    """Count number of frames in a scene"""
    frame_count = 1

    if sample["next"] != "":
        frame_count += 1
        sample_counter = nusc.get("sample", sample["next"])

        while sample_counter["next"] != "":
            frame_count += 1
            sample_counter = nusc.get("sample", sample_counter["next"])

    return frame_count

def load_2d_detections(gd_path):
    """Load 2D detection results"""
    with open(gd_path, "r") as f:
        results_GD = json.load(f)
    
    results_GD_new = dict()
    for result_GD in results_GD:
        file_name = result_GD['file_name']
        x1, y1, w, h = result_GD['bbox']
        x2, y2 = x1 + w, y1 + h    
        
        if file_name in results_GD_new:
            results_GD_new[file_name]['boxes_filt_list'].append([x1, y1, x2, y2])
            results_GD_new[file_name]['labels_list'].append([result_GD['category_id']-1])
            results_GD_new[file_name]['scores_list'].append([result_GD['score']])
        else:
            results_GD_new[file_name] = dict()
            results_GD_new[file_name]['boxes_filt_list'] = list()
            results_GD_new[file_name]['labels_list'] = list()
            results_GD_new[file_name]['scores_list'] = list()
            results_GD_new[file_name]['boxes_filt_list'].append([x1, y1, x2, y2])
            results_GD_new[file_name]['labels_list'].append([result_GD['category_id']-1])
            results_GD_new[file_name]['scores_list'].append([result_GD['score']])            
    
    return results_GD_new

def load_info_dict(nuscenes_root):
    """Load information dictionary"""
    info_path = nuscenes_root + '/nuscenes_infos_val.pkl'
    info_data = pickle.load(open(info_path, 'rb'))
    info_data = list(sorted(info_data['infos'], key=lambda e: e['timestamp']))
    
    info_dict = {}
    for frame_num, info in enumerate(info_data):
        if info['token'] not in info_dict:
            info_dict[info['token']] = {}
            info_dict[info['token']].update({
                'lidar2ego_translation': info['lidar2ego_translation'],
                'lidar2ego_rotation': info['lidar2ego_rotation'],
                'ego2global_translation': info['ego2global_translation'],
                'ego2global_rotation': info['ego2global_rotation'],
                'lidar_path': info['lidar_path'],
                'cams': info['cams']
            })
    
    return info_dict

def process_2d_detections(args, nusc, sample, results_GD_new, sample_token, sample_count=1):
    """Process 2D detection results"""   
    for view in args.views:
        if view not in VIEWS:
            print(f"Warning: View {view} is not in supported views list, skipping")
            continue
            
        try:
            (path, nusc_boxes, camera_instrinsic) = nusc.get_sample_data(sample["data"][view])
        except KeyError:
            print(f"Warning: No data for view {view} in sample {sample_token}, skipping")
            continue

        # Get 2D detection results
        file_name = path.split('/')[-1]
        if file_name in results_GD_new and args.use_gd:
            image_pil = Image.open(path).convert('RGB')
            image = np.array(image_pil)
            image = image[:, :, ::-1].copy()  
            boxes_filt_list = results_GD_new[file_name]['boxes_filt_list']
            labels_list = results_GD_new[file_name]['labels_list']
            scores_list = results_GD_new[file_name]['scores_list']
    
            # Draw 2D detection boxes
            for i, box in enumerate(boxes_filt_list):
                min_col, min_row, max_col, max_row = box
                label = labels_list[i][0]
                category_name = CUSTOM_VOCABULARY[label]
                score = scores_list[i][0]
                if score < args.score_threshold:
                    continue
                colors = nusc.colormap[category_name]
                # RGB -> BGR
                colors = (colors[2], colors[1], colors[0])
                cv2.rectangle(image, (int(min_col), int(min_row)), (int(max_col), int(max_row)), colors, 3) 
            
            # Save 2D detection results
            if sample_count % args.save_interval == 0:
                os.makedirs(f"{args.save_dir}/2D_detections/{view}", exist_ok=True)
                save_path = f"{args.save_dir}/2D_detections/{view}/{sample_token}&{view}_2D.jpg"
                cv2.imwrite(save_path, image)
               

def process_3d_detections(args, nusc, sample_token, info_dict, sample_count):
    """Process 3D detection results"""
    # Load 3D detection results
    res3d = json.load(open(args.res3d_path, 'r'))
    
    if sample_token not in res3d['results']:
        print(f"Warning: No 3D detection results for sample {sample_token}")
        return
    
    pred_res = res3d['results'][sample_token]
    box_dict = {'EGO': []}
    for view in VIEWS:
        box_dict[view] = []
    box_list = []
    
    for res in pred_res:
        score = res['detection_score']
        if score < args.score_threshold:
            continue
            
        pred_box_nusc = NuScenesBox(
            res['translation'],
            res['size'],
            Quaternion(res['rotation']),
            name=res['detection_name'],
            score=res['detection_score'],
            velocity=(0, 0, 0))
        
        # Transform to ego-pose coordinate system
        pred_box_nusc.translate(-np.array(info_dict[sample_token]['ego2global_translation']))
        pred_box_nusc.rotate(Quaternion(info_dict[sample_token]['ego2global_rotation']).inverse)
        box_dict['EGO'].append(pred_box_nusc)
        
        for view in VIEWS:
            pred_box_nusc_camera = pred_box_nusc.copy()
            pred_box_nusc_camera.translate(-np.array(info_dict[sample_token]['cams'][view]['sensor2ego_translation']))
            pred_box_nusc_camera.rotate(Quaternion(info_dict[sample_token]['cams'][view]['sensor2ego_rotation']).inverse)
            box_dict[view].append(pred_box_nusc_camera)
                        
        box_list.append(pred_box_nusc)
    
    # Save 3D detection results
    if sample_count % args.save_interval == 0:
        os.makedirs(f"{args.save_dir}/3D_only_lidar_detections", exist_ok=True)
        os.makedirs(f"{args.save_dir}/3D_detections", exist_ok=True)
        
        # Show only lidar detection results
        out_path = f"{args.save_dir}/3D_only_lidar_detections/{sample_token}_lidar.png"
        nusc.render_sample(True, False, sample_token, box_vis_level=BoxVisibility.ALL, 
                          out_path=out_path, box_list=box_list, box_dict=box_dict)
        
        # Show complete 3D detection results
        out_path = f"{args.save_dir}/3D_detections/{sample_token}.png"                
        nusc.render_sample(False, True, sample_token, box_vis_level=BoxVisibility.ALL, 
                          out_path=out_path, box_list=box_list, box_dict=box_dict)

def main():
    """Main function"""
    args = parse_args()
    
    # Print process ID
    getpid_X = os.getpid()
    print(f"Process ID: {getpid_X}")
    
    # Initialize NuScenes dataset
    nusc = NuScenes(version=args.version, dataroot=args.nuscenes_root, verbose=True)
    
    # Load info
    info_dict = load_info_dict(args.nuscenes_root)
    
    # Load 2D detection results
    if args.use_gd:
        results_GD_new = load_2d_detections(args.gd_path)
    else:
        results_GD_new = {}
    
    # Process samples
    sample_count = 1
    
    # If specific sample token is provided, process only that sample
    if args.sample_token:
        try:
            sample = nusc.get('sample', args.sample_token)
            print(f"Processing specific sample: {args.sample_token}")
            
            # Process 2D detections
            process_2d_detections(args, nusc, sample, results_GD_new, args.sample_token)
            
            # Process 3D detections
            process_3d_detections(args, nusc, args.sample_token, info_dict, sample_count)
            
        except Exception as e:
            print(f"Error processing sample {args.sample_token}: {e}")
        return
    
    # Process all validation set scenes
    for scene_num, scene_name in enumerate(val):
        scene_token = nusc.field2token("scene", "name", scene_name)[0]
        scene = nusc.get("scene", scene_token)
        sample = nusc.get("sample", scene["first_sample_token"])         
        num_frames = count_frames(nusc, sample)

        for f in range(num_frames):
            print(f'Processing sample {sample_count}/{len(info_dict)}')
            sample_token = sample["token"]
            
            # Process 2D detections
            process_2d_detections(args, nusc, sample, results_GD_new, sample_token, sample_count)
            
            # Process 3D detections
            process_3d_detections(args, nusc, sample_token, info_dict, sample_count)

            if sample['next'] != "":
                sample = nusc.get('sample', sample['next'])
                sample_count += 1
            else:
                break

if __name__ == '__main__':
    main()