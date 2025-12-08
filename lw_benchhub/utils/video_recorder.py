# Copyright 2025 Lightwheel Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import queue
import threading
from datetime import datetime
from pathlib import Path

import cv2
import mediapy as media
import numpy as np
import torch


class VideoRecorder:
    """Video recording utility class using mediapy."""

    def __init__(self, save_dir, fps=30, task=None, robot=None, layout=None):
        if layout and (layout.endswith("usd") or layout.endswith("usda")):
            layout = layout.split("/")[-1].split(".")[0]
        self.save_dir = Path(save_dir) / f"{layout}_{robot}_{task}"
        self.task = task
        self.robot = robot
        self.layout = layout
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.video_writer = None
        self.frame_count = 0

    def start_recording(self, camera_name, image_shape):
        """Start recording video"""
        self.frame_count = 0
        # close previous writers
        self.stop_recording()

        height, width = image_shape
        combined_shape = (height, width)

        # Generate timestamp in format YYYYMMDD_HHMM
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        video_filename = f"{timestamp}.mp4"
        video_path = self.save_dir / video_filename
        video_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            writer = media.VideoWriter(path=video_path, shape=combined_shape, fps=self.fps)
            writer.__enter__()  # manually enter context
            self.video_writer = writer
            print(f"✓ Successfully initialized combined recording")
            print(f"  Video filename: {video_filename}")
            print(f"  Combined shape: {combined_shape}")
        except Exception as e:
            print(f"✗ Failed to create combined VideoWriter: {e}")

    def add_frame(self, combined_image):
        """Add a combined frame to the video"""
        if self.video_writer is None:
            return
        try:
            frame = combined_image.cpu().numpy()
            self.video_writer.add_image(frame)
            self.frame_count += 1
            if self.frame_count % 100 == 0:
                print(f"Recorded {self.frame_count} combined frames")
        except Exception as e:
            print(f"Error adding frame: {e}")

    def stop_recording(self):
        """Stop recording and save video using mediapy"""
        if self.video_writer is not None:
            try:
                self.video_writer.__exit__(None, None, None)
                print(f"Save combined video completed, {self.frame_count} frames")
            except Exception as e:
                print(f"Error closing video writer: {e}")

        # Clear writer
        self.video_writer = None


def get_camera_images(env):
    """Get RGB images from all cameras and combine them horizontally"""
    camera_data = []
    camera_names = []
    try:
        # get camera images from observation manager
        obs = env.observation_manager.compute()
        for camera_name in [n for n, c in env.cfg.isaaclab_arena_env.embodiment.observation_cameras.items() if env.cfg.isaaclab_arena_env.task.task_type in c["tags"]]:
            if camera_name in obs.get('policy', {}):
                camera_data.append(obs['policy'][camera_name][0] if len(obs['policy'][camera_name].shape) == 4 else obs['policy'][camera_name])
                camera_names.append(camera_name)
        if len(camera_data) > 1:
            combined_image = _combine_camera_images(camera_data, camera_names)
            combined_name = "__".join(camera_names)
            return combined_image, combined_name
        elif len(camera_data) == 1:
            return camera_data[0], camera_names[0]
        else:
            return None, None
    except Exception as e:
        raise e


def calculate_camera_layout(num_cameras, max_cameras_per_row=4):
    """
    Calculate the grid layout for arranging camera images.

    Args:
        num_cameras: Total number of cameras
        max_cameras_per_row: Maximum number of cameras per row (default: 4)

    Returns:
        tuple: (num_rows, cameras_per_row_list, max_cameras_in_row)
            - num_rows: Number of rows needed
            - cameras_per_row_list: List of camera counts for each row
            - max_cameras_in_row: Maximum number of cameras in any row (for padding)
    """
    import math

    if num_cameras <= max_cameras_per_row:
        # Single row if total cameras <= max_cameras_per_row
        return 1, [num_cameras], num_cameras

    # Calculate minimum number of rows (m)
    m = math.ceil(num_cameras / max_cameras_per_row)

    # If n > 4*m, use m+1 rows, otherwise use m rows
    # When using m+1 rows, distribute according to n/(m+1)
    if num_cameras > max_cameras_per_row * m:
        num_rows = m + 1
    else:
        num_rows = m

    # Distribute cameras evenly across rows based on num_cameras / num_rows
    base_cameras_per_row = num_cameras // num_rows
    remainder = num_cameras % num_rows
    # Build list of cameras per row
    cameras_per_row_list = []
    for row_idx in range(num_rows):
        # First 'remainder' rows get one extra camera
        cameras_in_this_row = base_cameras_per_row + (1 if row_idx < remainder else 0)
        cameras_per_row_list.append(cameras_in_this_row)

    # Maximum cameras in any row (after padding, all rows will have this width)
    max_cameras_in_row = max(cameras_per_row_list)
    return num_rows, cameras_per_row_list, max_cameras_in_row


def _combine_camera_images(camera_data, camera_names):
    """Combine multiple camera images horizontally"""
    try:
        max_height = max(frame.shape[0] for frame in camera_data)
        padded_frames = []
        for frame in camera_data:
            current_height, current_width = frame.shape[:2]
            if current_height < max_height:
                padding_height = max_height - current_height
                if frame.dtype == torch.uint8:
                    padding = torch.full((padding_height, current_width, 3),
                                         255, dtype=torch.uint8, device=frame.device)
                else:
                    padding = torch.full((padding_height, current_width, 3),
                                         1.0, dtype=frame.dtype, device=frame.device)
                frame = torch.cat([frame, padding], dim=0)
            padded_frames.append(frame)
        combined_frame = torch.cat(padded_frames, dim=1)
        return combined_frame

    except Exception as e:
        print(f"Error combining camera images: {e}")
        import traceback
        traceback.print_exc()
        return None


class VideoProcessor:
    """Independent video processing thread for ordered frame handling"""

    def __init__(self, replay_mp4_path, video_height, video_width, args_cli, product_mp4_path=None, product_camera_names=None, product_video_height=0, product_video_width=0):
        self.replay_mp4_path = replay_mp4_path
        self.video_height = video_height
        self.video_width = video_width
        self.args_cli = args_cli
        self.product_mp4_path = product_mp4_path
        self.product_camera_names = product_camera_names or []
        self.product_video_height = product_video_height
        self.product_video_width = product_video_width
        self.frame_queue = queue.Queue(maxsize=100)
        self.running = True
        self.v = None
        self.product_v = None
        self.thread = threading.Thread(target=self._process_frames_worker, daemon=True)
        self.thread.start()

    def add_frame(self, obs, camera_names):
        """Add a frame to the processing queue"""
        if not self.running:
            return
        self.frame_queue.put_nowait((obs, camera_names))

    def _process_frames_worker(self):
        """Worker thread that processes frames in order"""
        self.v = media.VideoWriter(path=self.replay_mp4_path, shape=(self.video_height, self.video_width), fps=30)
        self.v.__enter__()

        # Initialize product video writer if product cameras are configured
        if self.product_mp4_path and self.product_camera_names and self.product_video_height > 0 and self.product_video_width > 0:
            self.product_v = media.VideoWriter(path=self.product_mp4_path, shape=(self.product_video_height, self.product_video_width), fps=30)
            self.product_v.__enter__()

        frame_count = 0
        product_frame_count = 0
        try:
            while self.running:
                if not self.frame_queue.empty():
                    obs, camera_names = self.frame_queue.get_nowait()
                    frame_count += 1
                    product_frames = self._process_single_frame(obs, camera_names)
                    product_frame_count += product_frames
                    self.frame_queue.task_done()
                else:
                    import time
                    time.sleep(0.01)

            # Process remaining frames after shutdown signal
            while not self.frame_queue.empty():
                obs, camera_names = self.frame_queue.get_nowait()
                frame_count += 1
                product_frames = self._process_single_frame(obs, camera_names)
                product_frame_count += product_frames
                self.frame_queue.task_done()

        except Exception as e:
            print(f"Video processing error: {e}")
        finally:
            if self.v:
                self.v.__exit__(None, None, None)
                print(f"Regular video processing completed, {frame_count} frames")
            if self.product_v and product_frame_count > 0:
                self.product_v.__exit__(None, None, None)
                print(f"Product video processing completed, {product_frame_count} frames")
            elif self.product_v:
                print("Product video skipped - no valid frames")

    def _process_single_frame(self, obs, camera_names):
        """Process a single frame"""
        camera_images = [obs[name].cpu().numpy() for name in camera_names]
        if not camera_images:
            return 0

        # Process regular cameras
        full_image = self._arrange_camera_images(camera_images)
        self.v.add_image(full_image)

        # Process product cameras (subset of regular cameras)
        product_frames = 0
        if self.product_v and self.product_camera_names:
            # Filter product cameras that are actually in camera_names
            available_product_cameras = [name for name in self.product_camera_names if name in camera_names]

            if available_product_cameras:
                product_camera_images = [obs['policy'][name].cpu().numpy() for name in available_product_cameras]
                if product_camera_images:
                    product_full_image = self._arrange_camera_images(product_camera_images)
                    self.product_v.add_image(product_full_image)
                    product_frames = 1

        if not self.args_cli.without_image:
            cv2.imshow("replay", full_image[..., ::-1])
            cv2.waitKey(1)

        return product_frames

    def _arrange_camera_images(self, camera_images):
        """Arrange camera images in a grid layout with balanced distribution"""
        camera_images = [img.squeeze(0) for img in camera_images]
        num_cameras = len(camera_images)

        max_cameras_per_row = 4  # Maximum cameras per row

        if num_cameras <= max_cameras_per_row:
            # Single row if total cameras <= 4
            full_image = np.concatenate(camera_images, axis=1)
        else:
            # Calculate layout using shared function
            num_rows, cameras_per_row_list, max_cameras_in_row = calculate_camera_layout(
                num_cameras, max_cameras_per_row
            )

            # Build rows
            rows = []
            camera_idx = 0
            for cameras_in_this_row in cameras_per_row_list:
                row = camera_images[camera_idx:camera_idx + cameras_in_this_row]
                camera_idx += cameras_in_this_row
                rows.append(row)

            # Process rows and pad to match max width
            row_images = []
            for row in rows:
                # Pad row with black image if needed to match max width
                if len(row) < max_cameras_in_row:
                    black_image = np.zeros_like(camera_images[0])
                    row.extend([black_image] * (max_cameras_in_row - len(row)))
                row_final = np.concatenate(row, axis=1)
                row_images.append(row_final)

            # Concatenate all rows vertically
            full_image = np.concatenate(row_images, axis=0)

        self.v.add_image(full_image)
        if not self.args_cli.without_image:
            cv2.imshow("replay", full_image[..., ::-1])
            cv2.waitKey(1)
        return full_image

    def shutdown(self):
        """Shutdown the video processor"""
        self.running = False
        import time
        start_time = time.time()
        while not self.frame_queue.empty() and time.time() - start_time < 10.0:
            time.sleep(0.1)

        try:
            self.frame_queue.join()
        except Exception:
            pass

        self.thread.join(timeout=3.0)

    def get_video_path(self):
        """Get the video file path"""
        return self.replay_mp4_path


def get_video_duration(video_path):
    """
    Get video duration in seconds

    Args:
        video_path (str): Path to the video file

    Returns:
        float: Video duration in seconds, or 0.0 if failed to read
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Unable to open video file: {video_path}")
            return 0.0

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        cap.release()

        if fps > 0:
            duration = frame_count / fps
            return round(duration, 2)
        else:
            return 0.0

    except Exception as e:
        print(f"Failed to obtain video duration: {e}")
        return 0.0
