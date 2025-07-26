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

        video_filename = f"{camera_name}.mp4"
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
        if hasattr(env.cfg, 'render_cfgs'):
            for camera_name in env.cfg.render_cfgs.keys():
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
