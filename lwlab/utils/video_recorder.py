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

import mediapy as media
import torch
from pathlib import Path
from datetime import datetime
import queue
import threading
import numpy as np
import cv2


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
        for camera_name in [n for n, c in env.cfg.isaac_arena_env.embodiment.observation_cameras.items() if env.cfg.isaac_arena_env.task.task_type in c["tags"]]:
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

    def __init__(self, replay_mp4_path, video_height, video_width, args_cli):
        self.replay_mp4_path = replay_mp4_path
        self.video_height = video_height
        self.video_width = video_width
        self.args_cli = args_cli
        self.frame_queue = queue.Queue(maxsize=100)
        self.running = True
        self.v = None
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

        frame_count = 0
        try:
            while self.running:
                if not self.frame_queue.empty():
                    obs, camera_names = self.frame_queue.get_nowait()
                    self._process_single_frame(obs, camera_names)
                    frame_count += 1
                    self.frame_queue.task_done()
                else:
                    import time
                    time.sleep(0.01)

            # Process remaining frames after shutdown signal
            while not self.frame_queue.empty():
                obs, camera_names = self.frame_queue.get_nowait()
                self._process_single_frame(obs, camera_names)
                frame_count += 1
                self.frame_queue.task_done()

        except Exception as e:
            print(f"Video processing error: {e}")
        finally:
            if self.v:
                self.v.__exit__(None, None, None)

    def _process_single_frame(self, obs, camera_names):
        """Process a single frame"""
        camera_images = [obs['camera_obs'][name].cpu().numpy() for name in camera_names]
        if not camera_images:
            return

        camera_images = [img.squeeze(0) for img in camera_images]
        num_cameras = len(camera_images)

        if num_cameras > 4:
            cameras_per_row = (num_cameras + 1) // 2
            first_row = camera_images[:cameras_per_row]
            first_row_final = np.concatenate(first_row, axis=1)
            second_row = camera_images[cameras_per_row:]
            if second_row:
                second_row_final = np.concatenate(second_row, axis=1)
                full_image = np.concatenate([first_row_final, second_row_final], axis=0)
            else:
                full_image = first_row_final
        else:
            full_image = np.concatenate(camera_images, axis=1)

        self.v.add_image(full_image)
        if not self.args_cli.without_image:
            cv2.imshow("replay", full_image[..., ::-1])
            cv2.waitKey(1)

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
