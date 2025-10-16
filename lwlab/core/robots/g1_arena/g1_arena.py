from isaac_arena.embodiments.g1.g1 import G1WBCJointEmbodiment, G1WBCPinkEmbodiment
from lwlab.core.robots.robot_arena_base import LwLabEmbodimentBase
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from lwlab.core.robots.unitree.g1 import G1_GEARWBC_CFG
from isaac_arena.utils.pose import Pose


class G1ArenaJointEmbodiment(G1WBCJointEmbodiment, LwLabEmbodimentBase):

    def __init__(self, enable_cameras: bool = False, initial_pose: Pose | None = None):
        super().__init__(enable_cameras, initial_pose)
        self.scene_config.robot = G1_GEARWBC_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.action_config.left_arm_action = None
        self.action_config.right_arm_action = None


class G1ArenaPinkEmbodiment(G1WBCPinkEmbodiment, LwLabEmbodimentBase):

    def __init__(self, enable_cameras: bool = False, initial_pose: Pose | None = None):
        super().__init__(enable_cameras, initial_pose)
        self.scene_config.robot = G1_GEARWBC_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.action_config.left_arm_action = None
        self.action_config.right_arm_action = None
