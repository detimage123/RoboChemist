"""
Script to convert AIRBOT raw data (from control_robot.py --record and the arm order is left,right, the cam order is front,left,right) to the LeRobot dataset v2.0 format.
Will zero pad non-exist data, so can't directly used by pi0 model.

Example usage: uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name> [--no-push-to-hub]
"""

import dataclasses
import json
import os
from pathlib import Path
import shutil
from typing import Literal

import cv2
import h5py
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub._download_raw import download_raw
import numpy as np
import tqdm
import tyro


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()

LEFT_INIT_POS = [
    -0.05664911866188049,
    -0.26874953508377075,
    0.5613412857055664,
    1.483367681503296,
    -1.1999313831329346,
    -1.3498512506484985,
    0,
]
RIGHT_INIT_POS = [
    -0.05664911866188049,
    -0.26874953508377075,
    0.5613412857055664,
    -1.483367681503296,
    1.1999313831329346,
    1.3498512506484985,
    0,
]
# TODO remove hardcoding
# we set the task name and replace it in AirBotInput during training
# task prompt will be its first match
TASKS = [
    "pick_place",
    "stack_block",
    "transfer_block",
    "stack_paper_cups",
    "fold_towel",
    "orgnize_block",
    "wipe_whiteboard",
]
TASKS = {name: name.upper() for name in TASKS}
RIGHT_ONLY_KEYS = ["pick_place"]
# INCLUDE_KEYS = ["pick_place"]
# INCLUDE_KEYS = ["stack_block"]
# INCLUDE_KEYS = ["orgnize_block"]
INCLUDE_KEYS = ["fold_towel"]
EXCLUDE_KEYS = [
    "pick_place_0116_yangz/47",
    # "stack_block_2/38", # 看起来没问题 不知道为什么在排除项里
    "stack_block_0104_tanner/0",  # 图片数量750多于JSON长度220
    # "stack_block_0105_yincheng/56", # 看起来没问题 不知道为什么在排除项里
    "stack_block_0105_yincheng/76",  # len(low_dim)=27
    "stack_block_0106_xuwang/7",  # 缺少low_dim.json
    "stack_paper_cups/0",  # 图片数量2103多于JSON长度1740
    "stack_paper_cups/5",  # 图片数量1656多于JSON长度1000
    "flatten_and_fold_towel/17",  # 图片数量2080多于JSON长度1155
    "flatten_and_fold_towel/35",  # 图片数量2110多于JSON长度1905
]
# 目前还不清楚图片数量更多的原因，为避免与low_dim.json不匹配先排除

def find_match(list, key):
    for item in list:
        if key in item:
            return item
    return None

def find_key(key_list, s):
    for key in key_list:
        if key in s:
            return key
    return None

def get_task_prompt(name):
    for task in TASKS:
        if task in name:
            return TASKS[task]
    raise ValueError(f"task not found in {name}")

def find_ep_dirs(dir_path):
    result = []

    for root, dirs, files in os.walk(dir_path):
        if INCLUDE_KEYS and not find_key(INCLUDE_KEYS, root):
            continue
        if "data_recording_info.json" in files:
            if not get_task_prompt(root):
                raise ValueError(f"skipped {root} not in TASKS")

            for dir_name in dirs:
                full_dir = str(Path(root) / dir_name)
                if EXCLUDE_KEYS and find_key(EXCLUDE_KEYS, full_dir):
                    print(f"skipped {full_dir} by EXCLUDE_KEYS")
                    continue
                result.append(full_dir)

    return sorted(result)


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    # in left-right order, refer to docs/norm_stats.md
    motors = [
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
        "left_gripper",
        "right_waist",
        "right_shoulder",
        "right_elbow",
        "right_forearm_roll",
        "right_wrist_angle",
        "right_wrist_rotate",
        "right_gripper",
    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        "right_only": {"dtype": "bool", "shape": (1,), "names": None},
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    for cam in ["cam_high", "cam_left_wrist", "cam_right_wrist"]:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    if Path(LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=25,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def get_cameras(hdf5_files: list[Path]) -> list[str]:
    with h5py.File(hdf5_files[0], "r") as ep:
        # ignore depth channel, not currently handled
        return [key for key in ep["/observations/images"].keys() if "depth" not in key]  # noqa: SIM118


def has_velocity(ep_paths) -> bool:
    with open(f"{ep_paths[0]}/low_dim.json") as f:
        low_dim = json.load(f)
    return "observation/arm/joint_velocity" in low_dim


def has_effort(ep_paths) -> bool:
    with open(f"{ep_paths[0]}/low_dim.json") as f:
        low_dim = json.load(f)
    return "observation/arm/joint_effort" in low_dim


def load_raw_episode_data(ep_path):
    imgs_dirs = os.listdir(ep_path)
    imgs_dirs = sorted([f"{ep_path}/{x}" for x in imgs_dirs if os.path.isdir(f"{ep_path}/{x}")])
    with open(f"{ep_path}/low_dim.json") as f:
        low_dim = json.load(f)
    ep_len = len(low_dim["action/arm/joint_position"])
    qpos = np.array(low_dim["observation/arm/joint_position"])
    qaction = np.array(low_dim["action/arm/joint_position"])
    gripper_pos = np.array(low_dim["observation/eef/joint_position"])
    gripper_action = np.array(low_dim["action/eef/joint_position"])

    if qpos.shape[-1] == 12:
        state = np.concatenate([qpos[:, :6], gripper_pos[:, 0:1], qpos[:, 6:], gripper_pos[:, 1:2]], axis=1)
        action = np.concatenate(
            [qaction[:, :6], gripper_action[:, 0:1], qaction[:, 6:], gripper_action[:, 1:2]], axis=1
        )
    elif qpos.shape[-1] == 6:
        # NOTE here assert single arm airbot datasets are right arm
        state = np.concatenate([np.tile(LEFT_INIT_POS, (ep_len, 1)), qpos, gripper_pos], axis=1)
        action = np.concatenate([np.tile(LEFT_INIT_POS, (ep_len, 1)), qaction, gripper_action], axis=1)
    else:
        raise ValueError

    velocity = None
    effort = None

    imgs_per_cam = {}
    for idx, camera in [(1, "cam_high"), (2, "cam_left_wrist"), (3, "cam_right_wrist")]:
        img_path = find_match(imgs_dirs, f"cam{idx}")
        if not img_path:
            imgs_per_cam[camera] = np.zeros_like(imgs_per_cam["cam_high"])
            continue
        imgs_array = []
        for i in range(ep_len):
            img = cv2.imread(f"{img_path}/frame_{i:06}.jpg", cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs_array.append(img)

        imgs_per_cam[camera] = np.array(imgs_array)

    return imgs_per_cam, state, action, velocity, effort


def populate_dataset(
    dataset: LeRobotDataset,
    ep_dirs,
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    if episodes is None:
        episodes = list(range(len(ep_dirs)))

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = ep_dirs[ep_idx]

        right_only = bool(find_key(RIGHT_ONLY_KEYS, ep_path))
        task_prompt = get_task_prompt(ep_path)

        try:
            imgs_per_cam, state, action, velocity, effort = load_raw_episode_data(ep_path)
            num_frames = state.shape[0]

            for i in range(num_frames):
                frame = {
                    "observation.state": state[i],
                    "action": action[i],
                    "right_only": right_only,
                }

                for camera, img_array in imgs_per_cam.items():
                    frame[f"observation.images.{camera}"] = img_array[i]

                if velocity is not None:
                    frame["observation.velocity"] = velocity[i]
                if effort is not None:
                    frame["observation.effort"] = effort[i]

                dataset.add_frame(frame)

            dataset.save_episode(task=task_prompt)

        except Exception as e:
            print(f"Error processing episode {ep_path}: {e}")
            dataset.clear_episode_buffer()

    return dataset


def port_aloha(
    raw_dir: Path,
    repo_id: str,
    raw_repo_id: str | None = None,
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = True,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    if (LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    if not raw_dir.exists():
        if raw_repo_id is None:
            raise ValueError("raw_repo_id must be provided if raw_dir does not exist")
        download_raw(raw_dir, repo_id=raw_repo_id)

    ep_dirs = find_ep_dirs(raw_dir)
    print(f"found {len(ep_dirs)} episodes.")

    dataset = create_empty_dataset(
        repo_id,
        robot_type="mobile_aloha" if is_mobile else "aloha",
        mode=mode,
        has_effort=has_effort(ep_dirs),
        has_velocity=has_velocity(ep_dirs),
        dataset_config=dataset_config,
    )
    dataset = populate_dataset(
        dataset,
        ep_dirs,
        episodes=episodes,
    )
    dataset.consolidate(run_compute_stats=False)

    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    tyro.cli(port_aloha)
