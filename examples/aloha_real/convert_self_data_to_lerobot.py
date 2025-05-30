"""
Script to convert Aloha hdf5 data to the LeRobot dataset v2.0 format.
Supports both episode_0.hdf5 and self.hdf5 file formats.
Now with support for reference images stored in a separate folder.

Example usage: uv run examples/aloha_real/convert_self_data_to_lerobot.py --raw-dir /path/to/raw/data --ref-image-dir /path/to/ref/images --repo-id <org>/<dataset-name>
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal, List, Optional
import os

import h5py
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub._download_raw import download_raw
import numpy as np
import torch
import tqdm
import tyro
import cv2


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None
    ref_image_dir: Path | None = None  # Directory containing reference images
    use_ref_image: bool = False  # Whether to include reference images in dataset


DEFAULT_DATASET_CONFIG = DatasetConfig()


def detect_cameras(hdf5_files: list[Path]) -> list[str]:
    """Detect available cameras in the HDF5 files."""
    cameras = []
    with h5py.File(hdf5_files[0], "r") as ep:
        if "/observations/images" in ep:
            # ignore depth channel, not currently handled
            cameras = [key for key in ep["/observations/images"].keys() if "depth" not in key]
    return cameras


def has_tp_image(hdf5_files: list[Path]) -> bool:
    """Check if HDF5 files contain tp_image data."""
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/tp_image" in ep


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    cameras: List[str],
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    has_tp_image: bool = False,
    has_ref_image: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    motors = [
        "right_waist",
        "right_shoulder",
        "right_elbow",
        "right_forearm_roll",
        "right_wrist_angle",
        "right_wrist_rotate",
        "right_gripper",
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
        "left_gripper",
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

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }
    
    # Add tp_image feature if available
    if has_tp_image:
        features["observation.tp_image"] = {
            "dtype": mode,
            "shape": (3, 480, 640),  # Assuming same dimensions as other images
            "names": [
                "channels",
                "height",
                "width",
            ],
        }
    
    # Add ref_image feature if available
    if has_ref_image:
        features["observation.ref_image"] = {
            "dtype": mode,
            "shape": (3, 480, 640),  # Assuming same dimensions as other images
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
        fps=50,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def has_velocity(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/qvel" in ep


def has_effort(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/effort" in ep


def load_raw_images_per_camera(ep: h5py.File, cameras: list[str]) -> dict[str, np.ndarray]:
    imgs_per_cam = {}
    for camera in cameras:
        if f"/observations/images/{camera}" not in ep:
            continue

        uncompressed = ep[f"/observations/images/{camera}"].ndim == 4

        if uncompressed:
            # load all images in RAM
            imgs_array = ep[f"/observations/images/{camera}"][:]
        else:
            import cv2

            # load one compressed image after the other in RAM and uncompress
            imgs_array = []
            for data in ep[f"/observations/images/{camera}"]:
                # Convert the HDF5 dataset to numpy array before decoding
                np_data = np.frombuffer(data.tobytes(), dtype=np.uint8)
                img = cv2.imdecode(np_data, 1)
                if img is not None:
                    imgs_array.append(img)
                else:
                    raise ValueError(f"Failed to decode image for camera {camera}")
            imgs_array = np.array(imgs_array)

        imgs_per_cam[camera] = imgs_array
    return imgs_per_cam


def load_tp_image(ep: h5py.File) -> Optional[np.ndarray]:
    """Load tp_image data if available."""
    if "/observations/tp_image" not in ep:
        return None
    
    tp_image_data = ep["/observations/tp_image"]
    
    # Check if images are compressed or not
    uncompressed = tp_image_data.ndim == 4
    
    if uncompressed:
        # load all images in RAM
        return tp_image_data[:]
    else:
        import cv2
        
        # load one compressed image after the other in RAM and uncompress
        imgs_array = []
        for data in tp_image_data:
            # Convert the HDF5 dataset to numpy array before decoding
            np_data = np.frombuffer(data.tobytes(), dtype=np.uint8)
            img = cv2.imdecode(np_data, 1)
            if img is not None:
                imgs_array.append(img)
            else:
                raise ValueError("Failed to decode tp_image")
        return np.array(imgs_array)


def load_ref_image(ref_image_dir: Path, episode_index: int, episode_filename: str) -> Optional[np.ndarray]:
    """
    Load reference image for an episode from the reference image directory.
    
    Args:
        ref_image_dir: Directory containing reference images
        episode_index: Index of the current episode
        episode_filename: Filename of the episode HDF5 file
        
    Returns:
        Reference image as numpy array or None if not found
    """
    if not ref_image_dir or not ref_image_dir.exists():
        return None
    
    # Extract episode number from filename for more accurate matching
    episode_number = None
    
    # Try to extract episode number from filename (like episode_1.hdf5 -> 1)
    import re
    match = re.search(r'episode_(\d+)\.hdf5', episode_filename)
    if match:
        episode_number = match.group(1)
    else:
        # If not in standard format, use the provided index
        episode_number = str(episode_index)
    
    # Look for matching PNG file
    ref_image_path = ref_image_dir / f"{episode_number}.png"
    
    if not ref_image_path.exists():
        print(f"Warning: Reference image {ref_image_path} not found for episode {episode_filename}")
        return None
    
    # Load and process the reference image
    try:
        # Read image using OpenCV
        img = cv2.imread(str(ref_image_path))
        if img is None:
            print(f"Warning: Failed to load reference image {ref_image_path}")
            return None
        
        # Convert from BGR to RGB (OpenCV loads as BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img
    except Exception as e:
        print(f"Error loading reference image {ref_image_path}: {e}")
        return None


def load_raw_episode_data(
    ep_path: Path,
    cameras: List[str],
    ref_image_dir: Optional[Path] = None,
    episode_index: int = 0,
) -> tuple[
    dict[str, np.ndarray],
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    try:
        with h5py.File(ep_path, "r") as ep:
            state = torch.from_numpy(ep["/observations/qpos"][:])
            action = torch.from_numpy(ep["/action"][:])

            velocity = None
            if "/observations/qvel" in ep:
                velocity = torch.from_numpy(ep["/observations/qvel"][:])

            effort = None
            if "/observations/effort" in ep:
                effort = torch.from_numpy(ep["/observations/effort"][:])

            imgs_per_cam = load_raw_images_per_camera(ep, cameras)
            
            # Load tp_image if available
            tp_image = load_tp_image(ep)
            
            # Get number of frames to prepare reference image
            num_frames = state.shape[0]

        # Load reference image if directory is provided
        ref_image = None
        if ref_image_dir is not None:
            single_ref_image = load_ref_image(ref_image_dir, episode_index, ep_path.name)
            if single_ref_image is not None:
                # Repeat the reference image for all frames in the episode
                ref_image = np.array([single_ref_image] * num_frames)

        return imgs_per_cam, state, action, velocity, effort, tp_image, ref_image
    except (OSError, IOError) as e:
        print(f"Error loading file {ep_path}: {e}")
        raise


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    cameras: List[str],
    task: str,
    episodes: list[int] | None = None,
    ref_image_dir: Optional[Path] = None,
) -> LeRobotDataset:
    if episodes is None:
        episodes = range(len(hdf5_files))

    for ep_idx in tqdm.tqdm(episodes):
        if ep_idx >= len(hdf5_files):
            break

        ep_path = hdf5_files[ep_idx]
        
        try:
            imgs_per_cam, state, action, velocity, effort, tp_image, ref_image = load_raw_episode_data(
                ep_path, 
                cameras, 
                ref_image_dir=ref_image_dir, 
                episode_index=ep_idx
            )
            
            num_frames = state.shape[0]

            for i in range(num_frames):
                frame = {
                    "observation.state": state[i],
                    "action": action[i],
                }

                for camera, img_array in imgs_per_cam.items():
                    frame[f"observation.images.{camera}"] = img_array[i]

                if velocity is not None:
                    frame["observation.velocity"] = velocity[i]
                if effort is not None:
                    frame["observation.effort"] = effort[i]
                if tp_image is not None:
                    frame["observation.tp_image"] = tp_image[i]
                if ref_image is not None:
                    frame["observation.ref_image"] = ref_image[i]

                dataset.add_frame(frame)

            dataset.save_episode(task=task)
        except Exception as e:
            print(f"Failed to process episode {ep_idx}, file {ep_path}: {e}")
            print(f"Skipping episode {ep_idx} and continuing with the next one.")
            continue

    return dataset


def check_hdf5_integrity(hdf5_files: list[Path]) -> list[int]:
    """
    Check the integrity of all HDF5 files and return indices of corrupted files.
    
    Args:
        hdf5_files: List of HDF5 file paths to check
        
    Returns:
        List of indices of corrupted files
    """
    corrupted_indices = []
    
    print(f"Checking integrity of {len(hdf5_files)} HDF5 files...")
    for i, file_path in enumerate(tqdm.tqdm(hdf5_files)):
        try:
            with h5py.File(file_path, "r") as f:
                # Try to access key datasets to verify file integrity
                _ = f["/observations/qpos"].shape
                _ = f["/action"].shape
        except Exception as e:
            print(f"File {i}: {file_path.name} is corrupted: {e}")
            corrupted_indices.append(i)
    
    if corrupted_indices:
        print(f"Found {len(corrupted_indices)} corrupted files at indices: {corrupted_indices}")
    else:
        print("All files are intact!")
        
    return corrupted_indices


def check_ref_images(ref_image_dir: Path, hdf5_files: list[Path]) -> tuple[bool, list[int]]:
    """
    Check if reference image directory exists and contains matching images for HDF5 files.
    
    Args:
        ref_image_dir: Directory containing reference images
        hdf5_files: List of HDF5 file paths
        
    Returns:
        Tuple of (has_any_images, missing_indices)
    """
    if not ref_image_dir or not ref_image_dir.exists():
        print(f"Reference image directory {ref_image_dir} does not exist")
        return False, list(range(len(hdf5_files)))
    
    print(f"Checking for reference images in {ref_image_dir}...")
    
    missing_indices = []
    has_any_images = False
    
    import re
    for i, hdf5_path in enumerate(hdf5_files):
        # Extract episode number from filename
        episode_number = None
        match = re.search(r'episode_(\d+)\.hdf5', hdf5_path.name)
        if match:
            episode_number = match.group(1)
        else:
            # If not in standard format, use the index
            episode_number = str(i)
        
        # Check if matching PNG exists
        ref_image_path = ref_image_dir / f"{episode_number}.png"
        if not ref_image_path.exists():
            missing_indices.append(i)
        else:
            has_any_images = True
    
    if missing_indices:
        print(f"Missing reference images for {len(missing_indices)} HDF5 files")
        if len(missing_indices) < 10:
            missing_files = [hdf5_files[idx].name for idx in missing_indices]
            print(f"Missing reference images for: {missing_files}")
        else:
            missing_files = [hdf5_files[idx].name for idx in missing_indices[:10]]
            print(f"First 10 missing reference images for: {missing_files}...")
    else:
        print("All reference images are available!")
    
    return has_any_images, missing_indices


def port_aloha(
    raw_dir: Path,
    repo_id: str,
    raw_repo_id: str | None = None,
    task: str = "DEBUG",
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = False,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    check_only: bool = False,
):
    if (LEROBOT_HOME / repo_id).exists() and not check_only:
        shutil.rmtree(LEROBOT_HOME / repo_id)

    if not raw_dir.exists():
        if raw_repo_id is None:
            raise ValueError("raw_repo_id must be provided if raw_dir does not exist")
        download_raw(raw_dir, repo_id=raw_repo_id)

    # Support both episode_*.hdf5 and self.hdf5 patterns
    hdf5_files = sorted(list(raw_dir.glob("*.hdf5")))
    
    if not hdf5_files:
        raise ValueError(f"No HDF5 files found in {raw_dir}")

    # Check file integrity first
    corrupted_indices = check_hdf5_integrity(hdf5_files)
    
    # Check reference images if enabled and directory is provided
    has_ref_images = False
    if dataset_config.use_ref_image:
        if dataset_config.ref_image_dir is None:
            raise ValueError("ref_image_dir must be provided when use_ref_image=True")
        if not dataset_config.ref_image_dir.exists():
            raise ValueError(f"Reference image directory '{dataset_config.ref_image_dir}' does not exist")
        
        # Check that all HDF5 files have corresponding PNG files
        has_ref_images, missing_indices = check_ref_images(dataset_config.ref_image_dir, hdf5_files)
        
        if not has_ref_images:
            raise ValueError(f"No reference images found in '{dataset_config.ref_image_dir}' that match the HDF5 files")
        
        # If any PNG files are missing, report error with missing indices
        if missing_indices:
            if len(missing_indices) <= 10:
                missing_files = [hdf5_files[idx].name for idx in missing_indices]
                raise ValueError(f"Missing reference images for HDF5 files: {missing_files}")
            else:
                missing_files = [hdf5_files[idx].name for idx in missing_indices[:10]]
                raise ValueError(f"Missing reference images for {len(missing_indices)} HDF5 files, including: {missing_files}...")
    
    # If check_only flag is set, stop here after checking
    if check_only:
        print("Integrity check complete. Exiting without data conversion.")
        return
    
    # Remove corrupted files from processing
    valid_episodes = None
    if episodes is None:
        # Create list of all valid episode indices
        valid_episodes = [i for i in range(len(hdf5_files)) if i not in corrupted_indices]
    else:
        # Filter provided episodes list to remove corrupted ones
        valid_episodes = [ep for ep in episodes if ep not in corrupted_indices]
    
    print(f"Proceeding with {len(valid_episodes)} valid episodes...")
    
    # Detect available cameras from valid files
    valid_file_for_detection = hdf5_files[valid_episodes[0]] if valid_episodes else None
    if not valid_file_for_detection:
        raise ValueError("No valid HDF5 files found for processing")
    
    # Detect available cameras
    cameras = detect_cameras([valid_file_for_detection])
    
    # Check if tp_image is available
    has_tp_img = has_tp_image([valid_file_for_detection])

    dataset = create_empty_dataset(
        repo_id,
        robot_type="mobile_aloha" if is_mobile else "aloha",
        cameras=cameras,
        mode=mode,
        has_effort=has_effort([valid_file_for_detection]),
        has_velocity=has_velocity([valid_file_for_detection]),
        has_tp_image=has_tp_img,
        has_ref_image=has_ref_images,
        dataset_config=dataset_config,
    )
    
    dataset = populate_dataset(
        dataset,
        hdf5_files,
        cameras=cameras,
        task=task,
        episodes=valid_episodes,
        ref_image_dir=dataset_config.ref_image_dir if has_ref_images else None,
    )
    dataset.consolidate()

    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    # Add a custom command to only check file integrity
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        # Remove the "check" argument
        sys.argv.pop(1)
        # Add check_only=True to args
        sys.argv.append("--check-only=True")
    
    # Example of how to use with reference images:
    # To use reference images:
    # --dataset-config.ref_image_dir=/path/to/ref/images --dataset-config.use_ref_image=True
    
    tyro.cli(port_aloha)