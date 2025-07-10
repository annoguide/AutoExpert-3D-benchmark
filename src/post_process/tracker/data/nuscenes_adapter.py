import math
from copy import deepcopy
from typing import Dict, List

import numpy as np
from nuscenes import NuScenes
from pyquaternion import Quaternion

from tracker.data.utils import progressbar, wrap_pi


def unpack_and_annotate_labels(
    labels: List[Dict], nusc: NuScenes, classes: List[str]
) -> List[Dict]:
    """
    Returns
    -------
    list:
    """
    timestamps = [frame["timestamp"] for frame in labels]
    assert sorted(timestamps) == timestamps, "label frames are not sorted by timestamp"

    '''Using a part of annotations'''
    # labels = labels[:12000]


    print("converting detections not in our expected classes to the OTHER class")
    unpacked_labels = []
    for label in progressbar(labels, "annotating labels"):
        sample = nusc.get("sample", label["token"])
        track_ids = np.array(
            [nusc.get("sample_annotation", t)["instance_token"] for t in sample["anns"]]
        )
        assert len(track_ids) == len(
            label["gt_names"]
        ), "number of boxes from nuscenes does not match the labels"

        bboxes = np.array(label["gt_boxes"])
        # label all unknown classes as "OTHER"
        names = np.array([n if n in classes else "OTHER" for n in label["gt_names"]])
        classes = [*classes, "OTHER"]
        unpacked_labels.append(
            {
                "translation": bboxes[:, :3],
                "size": bboxes[:, 3:6],
                "yaw": wrap_pi(bboxes[:, 6]),
                "velocity": np.array(label["gt_velocity"]),
                "label": np.array([classes.index(n) for n in names], dtype=int),
                "name": names,
                "track_id": track_ids,
                "timestamp_ns": label["timestamp"] * 1000,
                "lidar2ego_rotation": label["lidar2ego_rotation"],
                "lidar2ego_translation": label["lidar2ego_translation"],
                "ego2global_rotation": label["ego2global_rotation"],
                "ego2global_translation": label["ego2global_translation"],
                "frame_id": label["token"],
                "lidar_path": label["lidar_path"],
                "seq_id": sample["scene_token"],
            }
        )
    return unpacked_labels


def transform_to_reference_pose(detections: Dict[str, List[Dict]]):
    detections = deepcopy(detections)
    for _, frames in detections.items():
        for detection in frames:
            # transform xyz to global reference frame
            lidar2ego_rot = Quaternion(detection["lidar2ego_rotation"]).rotation_matrix
            ego2global_rot = Quaternion(
                detection["ego2global_rotation"]
            ).rotation_matrix
            detection["translation"] = detection[
                "translation"
            ] @ lidar2ego_rot.T + np.array(detection["lidar2ego_translation"])
            detection["translation"] = detection[
                "translation"
            ] @ ego2global_rot.T + np.array(detection["ego2global_translation"])

            # transform velocity, yaw to global reference frame
            rotation = ego2global_rot @ lidar2ego_rot
            velocity_3d = np.pad(
                detection["velocity"], [(0, 0), (0, 1)]
            )  # pad last dimension -> [x, y, 0]
            detection["velocity"] = (velocity_3d @ rotation.T)[:, :2]  # I, 2
            ego_to_city_yaw = math.atan2(rotation[1, 0], rotation[0, 0])
            detection["yaw"] = wrap_pi(detection["yaw"] + ego_to_city_yaw)  # I

    return detections

def transform_to_lidar(xyz, detection):
    # transform xyz to lidar reference frame
    lidar2ego_rot = Quaternion(detection["lidar2ego_rotation"]).rotation_matrix
    ego2lidar_rot = np.linalg.inv(lidar2ego_rot)
    
    ego2global_rot = Quaternion(
        detection["ego2global_rotation"]
    ).rotation_matrix
    global2ego_rot = np.linalg.inv(ego2global_rot)
    
    xyz = global2ego_rot @ (xyz - np.array(detection["ego2global_translation"]))
    xyz = ego2lidar_rot @ (xyz - np.array(detection["lidar2ego_translation"]))

    return xyz