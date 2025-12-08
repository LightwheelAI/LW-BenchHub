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

import asyncio
import traceback
from multiprocessing import Array, Process, Value, shared_memory

import numpy as np

from vuer import Vuer
from vuer.schemas import DefaultScene, Hands, MotionControllers, WebRTCStereoVideoPlane
# from webrtc.zed_server import *


class OpenTeleVision:
    def __init__(self, img_shape, shm_name, device_type, stream_mode="image", cert_file="./cert.pem", key_file="./key.pem", ngrok=True):
        # device_type: "controller" or "hand"
        self.device_type = device_type
        self.img_shape = (img_shape[0], img_shape[1], 3)
        self.img_height, self.img_width = img_shape[:2]
        self.img_width = self.img_width // 2

        self.shm_name = shm_name
        self.stream_mode = stream_mode
        self.cert_file = cert_file
        self.key_file = key_file
        self.ngrok = ngrok

        self.left_hand_shared = Array('d', 16, lock=True)
        self.right_hand_shared = Array('d', 16, lock=True)
        self.left_landmarks_shared = Array('d', 75, lock=True)
        self.right_landmarks_shared = Array('d', 75, lock=True)
        self.left_controller_state_shared = Array('d', 7, lock=True)  # trigger, squeeze, thumbstick_x, thumbstick_y, thumbstick, a_button, b_button
        self.right_controller_state_shared = Array('d', 7, lock=True)  # trigger, squeeze, thumbstick_x, thumbstick_y, thumbstick, a_button, b_button

        self.head_matrix_shared = Array('d', 16, lock=True)
        self.aspect_shared = Value('d', 1.0, lock=True)

        self.process = Process(target=self.run)
        self.process.daemon = True
        self.process.start()
        # self.run()

    def run(self):
        # self.loop = asyncio.new_event_loop()
        # asyncio.set_event_loop(self.loop)
        if self.ngrok:
            self.app = Vuer(host='0.0.0.0', queries=dict(grid=False), queue_len=3)
        else:
            self.app = Vuer(host='0.0.0.0', cert=self.cert_file, key=self.key_file, queries=dict(grid=False), queue_len=3)

        if self.device_type == "hand":
            self.app.add_handler("HAND_MOVE")(self.on_hand_move)
        elif self.device_type == "controller":
            self.app.add_handler("CONTROLLER_MOVE")(self.on_motion_controller_move)
        else:
            raise ValueError("device_type must be either 'hand' or 'controller'")
        self.app.add_handler("CAMERA_MOVE")(self.on_cam_move)

        if self.stream_mode == "image":
            existing_shm = shared_memory.SharedMemory(name=self.shm_name)
            self.img_array = np.ndarray((self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8, buffer=existing_shm.buf)
            self.app.spawn(start=True)(self.main_image)
        elif self.stream_mode == "webrtc":
            self.app.spawn(start=True)(self.main_webrtc)
        else:
            raise ValueError("stream_mode must be either 'webrtc' or 'image'")
        # self.app.run()

    def close(self):
        # self.app.close_ws()
        # loop=asyncio.get_event_loop()
        print("closing tv")

        async def shutdown():
            # close event loop
            self.loop.stop()
        # self.loop.call_soon_threadsafe(lambda: asyncio.ensure_future(shutdown()))
        self.process.kill()

    async def on_cam_move(self, event, session, fps=60):
        # print(f"head move {time.perf_counter():.3f}", event)
        # only intercept the ego camera.
        # if event.key != "ego":
        #     return
        try:
            # with self.head_matrix_shared.get_lock():  # Use the lock to ensure thread-safe updates
            #     self.head_matrix_shared[:] = event.value["camera"]["matrix"]
            # with self.aspect_shared.get_lock():
            #     self.aspect_shared.value = event.value['camera']['aspect']
            self.head_matrix_shared[:] = event.value["camera"]["matrix"]
            self.aspect_shared.value = event.value['camera']['aspect']
        except Exception as e:
            print(f"on cam move error: {e}. event.value=\n{event.value}")
        # self.head_matrix = np.array(event.value["camera"]["matrix"]).reshape(4, 4, order="F")
        # print(np.array(event.value["camera"]["matrix"]).reshape(4, 4, order="F"))
        # print("camera moved", event.value["matrix"].shape, event.value["matrix"])

    async def on_motion_controller_move(self, event, session, fps=60):
        # print(f"motion controller move {time.perf_counter():.3f}", fps, event, session)
        if "right" in event.value:
            self._parse_hand_data(event.value, "right", self.right_hand_shared)
            self._parse_controller_state(event.value, "right", self.right_controller_state_shared)
        if "left" in event.value:
            self._parse_hand_data(event.value, "left", self.left_hand_shared)
            self._parse_controller_state(event.value, "left", self.left_controller_state_shared)

    def _parse_controller_state(self, value, side, controller_state_shared):
        value = value[f"{side}State"]
        if len(value) == 0:
            return
        # {'trigger': False, 'squeeze': False, 'touchpad': False, 'thumbstick': False, 'aButton': False, 'bButton': False, 'triggerValue': 0, 'squeezeValue': 0, 'touchpadValue': [0, 0], 'thumbstickValue': [0, 0], 'aButtonValue': False, 'bButtonValue': False}
        values = [
            float(value["triggerValue"]),  # 0 to 1
            float(value["squeezeValue"]),  # 0 to 1
            -float(value["thumbstickValue"][1]),  # -1 to 1
            -float(value["thumbstickValue"][0]),  # -1 to 1
            float(value["thumbstick"]),  # 0 or 1
            float(value["aButtonValue"]),  # 0 or 1
            float(value["bButtonValue"]),  # 0 or 1
        ]
        controller_state_shared[:] = np.array(values)

    def _parse_hand_data(self, value, side, hand_shared, landmarks_shared=None):
        data = value[side]
        if not isinstance(data, list):
            # when the hand is not detected, the data is not a list.
            return
        data = np.array(data)
        hand_shared[:] = data[:16]
        if landmarks_shared is not None:
            data = data.reshape(25, 4, 4).transpose(0, 2, 1)
            landmarks_shared[:] = data[:, :3, 3].flatten()  # only use the position
        # if side == "right":
        #     print(f"[DEBUG] wrist landmarks: {data[0, :3, 3]}")

    async def on_hand_move(self, event, session, fps=60):
        # print(f"hand move {time.perf_counter():.3f}", fps, event)
        # with self.left_hand_shared.get_lock():  # Use the lock to ensure thread-safe updates
        #     self.left_hand_shared[:] = event.value["leftHand"]
        # with self.right_hand_shared.get_lock():
        #     self.right_hand_shared[:] = event.value["rightHand"]
        # with self.left_landmarks_shared.get_lock():
        #     self.left_landmarks_shared[:] = np.array(event.value["leftLandmarks"]).flatten()
        # with self.right_landmarks_shared.get_lock():
        #     self.right_landmarks_shared[:] = np.array(event.value["rightLandmarks"]).flatten()
        # self.left_hand_shared[:] = event.value["leftHand"]
        # self.right_hand_shared[:] = event.value["rightHand"]
        # self.left_landmarks_shared[:] = np.array(event.value["leftLandmarks"]).flatten()
        # self.right_landmarks_shared[:] = np.array(event.value["rightLandmarks"]).flatten()
        try:
            self._parse_hand_data(event.value, "left", self.left_hand_shared, self.left_landmarks_shared)
        except Exception as e:
            traceback.print_exc()
            print(f"on left hand move error: {e}. event.value=\n{event.value}")
        try:
            self._parse_hand_data(event.value, "right", self.right_hand_shared, self.right_landmarks_shared)
        except Exception as e:
            traceback.print_exc()
            print(f"on right hand move error: {e}. event.value=\n{event.value}")

    async def main_webrtc(self, session, fps=60):
        session.set @ DefaultScene(frameloop="always")
        session.upsert @ Hands(fps=fps, stream=True, key="hands", showLeft=False, showRight=False)
        session.upsert @ WebRTCStereoVideoPlane(
            src="https://192.168.8.102:8080/offer",
            key="zed",
            aspect=1.33334,
            height=8,
            position=[0, -2, -0.2],
        )
        while True:
            await asyncio.sleep(1)

    async def main_image(self, session, fps=60):
        if self.device_type == "hand":
            session.upsert @ Hands(fps=fps, stream=True, key="hands")
        elif self.device_type == "controller":
            session.upsert @ MotionControllers(stream=True, key="motion-controller-right", right=True)
            session.upsert @ MotionControllers(stream=True, key="motion-controller-left", left=True)  # Two controllers are only supported on PICO, Quest will get browser frozen.
        while True:
            # aspect = self.aspect_shared.value
            # display_image = self.img_array

            # session.upsert(
            # ImageBackground(
            #     # Can scale the images down.
            #     display_image[:self.img_height],
            #     # 'jpg' encoding is significantly faster than 'png'.
            #     format="jpeg",
            #     quality=80,
            #     key="left-image",
            #     interpolate=True,
            #     # fixed=True,
            #     aspect=1.778,
            #     distanceToCamera=2,
            #     position=[0, -0.5, -2],
            #     rotation=[0, 0, 0],
            # ),
            # to="bgChildren",
            # )

            # session.upsert(
            #     [
            #         ImageBackground(
            #             display_image[:, :self.img_width],
            #             aspect=1.778,
            #             height=1,
            #             distanceToCamera=1,
            #             # The underlying rendering engine supported a layer binary bitmask for both objects and the camera.
            #             # Below we set the two image planes, left and right, to layers=1 and layers=2.
            #             # Note that these two masks are associated with left eye's camera and the right eye's camera.
            #             layers=1,
            #             format="jpeg",
            #             quality=50,
            #             key="background-left",
            #             interpolate=True,
            #         ),
            #         ImageBackground(
            #             display_image[:, self.img_width:],
            #             aspect=1.778,
            #             height=1,
            #             distanceToCamera=1,
            #             layers=2,
            #             format="jpeg",
            #             quality=50,
            #             key="background-right",
            #             interpolate=True,
            #         ),
            #     ],
            #     to="bgChildren",
            # )
            await asyncio.sleep(0.03)

    @property
    def left_hand(self):
        # with self.left_hand_shared.get_lock():
        #     return np.array(self.left_hand_shared[:]).reshape(4, 4, order="F")
        return np.array(self.left_hand_shared[:]).reshape(4, 4, order="F")

    @property
    def right_hand(self):
        # with self.right_hand_shared.get_lock():
        #     return np.array(self.right_hand_shared[:]).reshape(4, 4, order="F")
        return np.array(self.right_hand_shared[:]).reshape(4, 4, order="F")

    @property
    def left_landmarks(self):
        # with self.left_landmarks_shared.get_lock():
        #     return np.array(self.left_landmarks_shared[:]).reshape(25, 3)
        return np.array(self.left_landmarks_shared[:]).reshape(25, 3)

    @property
    def right_landmarks(self):
        # with self.right_landmarks_shared.get_lock():
        # return np.array(self.right_landmarks_shared[:]).reshape(25, 3)
        return np.array(self.right_landmarks_shared[:]).reshape(25, 3)

    @property
    def head_matrix(self):
        # with self.head_matrix_shared.get_lock():
        #     return np.array(self.head_matrix_shared[:]).reshape(4, 4, order="F")
        return np.array(self.head_matrix_shared[:]).reshape(4, 4, order="F")

    @property
    def left_controller_state(self):
        trigger, squeeze, thumbstick_x, thumbstick_y, thumbstick, a_button, b_button = self.left_controller_state_shared
        return {
            "trigger": trigger,
            "squeeze": squeeze,
            "thumbstick_x": thumbstick_x,
            "thumbstick_y": thumbstick_y,
            "thumbstick": bool(thumbstick),
            "a_button": bool(a_button),
            "b_button": bool(b_button),
        }

    @property
    def right_controller_state(self):
        trigger, squeeze, thumbstick_x, thumbstick_y, thumbstick, a_button, b_button = self.right_controller_state_shared
        return {
            "trigger": trigger,
            "squeeze": squeeze,
            "thumbstick_x": thumbstick_x,
            "thumbstick_y": thumbstick_y,
            "thumbstick": bool(thumbstick),
            "a_button": bool(a_button),
            "b_button": bool(b_button),
        }

    @property
    def aspect(self):
        # with self.aspect_shared.get_lock():
        # return float(self.aspect_shared.value)
        return float(self.aspect_shared.value)


# if __name__ == "__main__":
#     resolution = (720, 1280)
#     crop_size_w = 340  # (resolution[1] - resolution[0]) // 2
#     crop_size_h = 270
#     resolution_cropped = (resolution[0] - crop_size_h, resolution[1] - 2 * crop_size_w)  # 450 * 600
#     img_shape = (2 * resolution_cropped[0], resolution_cropped[1], 3)  # 900 * 600
#     img_height, img_width = resolution_cropped[:2]  # 450 * 600
#     shm = shared_memory.SharedMemory(create=True, size=np.prod(img_shape) * np.uint8().itemsize)
#     shm_name = shm.name
#     img_array = np.ndarray((img_shape[0], img_shape[1], 3), dtype=np.uint8, buffer=shm.buf)

#     tv = OpenTeleVision(resolution_cropped, cert_file="../cert.pem", key_file="../key.pem")
#     while True:
#         # print(tv.left_landmarks)
#         # print(tv.left_hand)
#         # tv.modify_shared_image(random=True)
#         time.sleep(1)
