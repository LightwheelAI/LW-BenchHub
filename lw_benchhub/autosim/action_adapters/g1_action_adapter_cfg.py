from isaaclab.utils import configclass

from autosim import ActionAdapterCfg

from .g1_action_adapter import G1ActionAdapter


@configclass
class G1ActionAdapterCfg(ActionAdapterCfg):
    """Configuration for the G1 action adapter."""

    class_type: type = G1ActionAdapter

    ee_link_name: str = ""
    """End-effector link name, used to determine active hand (left/right)."""

    finger_close_angles: tuple = (-1.2, -1.2, -1.2, -1.2, -1.0, -1.0, -1.0, -1.2, -1.2, -1.2, -1.2, -1.0, -1.0, -1.0)
    """Per-joint finger angles (rad) when gripper is closed (default fallback)."""
    finger_open_angles: tuple = (0.0,) * 14
    """Per-joint finger angles (rad) when gripper is open."""
    skill_finger_configs: dict[str, dict[str, tuple]] | None = None
    """Per-skill per-hand finger configs. Format: {"left_hand": {"lift": (7 vals), ...}, "right_hand": {...}}"""
