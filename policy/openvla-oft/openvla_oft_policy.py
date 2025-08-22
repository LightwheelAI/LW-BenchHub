

from third_party.openvla_oft.experiments.robot.robot_utils import get_model


class OpenVLAOFTPolicy():
    def __init__(self, cfg):
        self.cfg = cfg
        self.model, self.action_head, self.proprio_projector, self.noisy_action_projector, self.processor = get_model(cfg)
        self.task_description = cfg.task_description
