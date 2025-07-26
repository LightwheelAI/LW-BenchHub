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

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm

from lwlab.utils.env import set_camera_follow_pose
from lwlab.utils.isaaclab_utils.assets import GeneralAssetCfg
from lwlab.core import LwBaseCfg


@configclass
class USDSceneCfg(InteractiveSceneCfg):
    """Configuration for the kitchen scene with a robot and a kitchen.

    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the robot and end-effector frames
    """
    _usd_path: str = MISSING
    main_scene: GeneralAssetCfg = None  # will be populated at __post_init__

    def __post_init__(self):
        super().__post_init__()

        self.main_scene = GeneralAssetCfg(
            prim_path="{ENV_REGEX_NS}/Scene",
            # Make sure to set the correct path to the generated scene
            spawn=sim_utils.UsdFileCfg(usd_path=self._usd_path, activate_contact_sensors=True),
        )

    # robots, Will be populated by agent env cfg
    robot: ArticulationCfg = MISSING

    # lights
    light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1500.0),
    )


# monkey patch to add the _usd_path field to the InteractiveSceneCfg class,
# so that InteractiveSceneCfg._add_entities_from_cfg can ignore _usd_path
InteractiveSceneCfg.__dataclass_fields__["_usd_path"] = USDSceneCfg.__dataclass_fields__["_usd_path"]


# @configclass
class BaseSceneEnvCfg(LwBaseCfg):
    """Configuration for the kitchen environment."""

    # Scene settings
    scene: USDSceneCfg = MISSING
    usd_path: str = MISSING

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        self.scene = USDSceneCfg(
            num_envs=4096,
            env_spacing=10.0,
            _usd_path=self.usd_path
        )
        if hasattr(self, "enable_cameras") and self.enable_cameras == True and hasattr(self, "render_cfgs"):
            render_resolution = None
            if hasattr(self, "replay_cfgs") and self.replay_cfgs.get("render_resolution", None) is not None:
                render_resolution = self.replay_cfgs["render_resolution"]
            for name, camera_cfg in self.render_cfgs.items():
                if render_resolution is not None:
                    camera_cfg.width = render_resolution[0]
                    camera_cfg.height = render_resolution[1]
                setattr(self.scene, name, camera_cfg)

        # general settings
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.render.enable_translucency = True

        def camera_pose_update(env):
            set_camera_follow_pose(env, self.viewport_cfg["offset"], self.viewport_cfg["lookat"])
            return False
        if self.first_person_view:
            self.terminations.camera_pose_update = DoneTerm(func=camera_pose_update)

    def _reset_internal(self, env_ids):
        pass

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["usd_path"] = self.usd_path
        return ep_meta
