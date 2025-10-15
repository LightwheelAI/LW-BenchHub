from isaac_arena.assets.object import Object
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaac_arena.assets.object_base import ObjectType
import isaaclab.sim as sim_utils


class LwLabObject(Object):

    def _generate_rigid_cfg(self) -> RigidObjectCfg:
        assert self.object_type == ObjectType.RIGID
        object_cfg = RigidObjectCfg(
            prim_path=self.prim_path,
            spawn=UsdFileCfg(
                usd_path=self.usd_path,
                scale=self.scale,
                activate_contact_sensors=False,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                sleep_threshold=0.0,
                stabilization_threshold=0.0,
            ),
        )
        object_cfg = self._add_initial_pose_to_cfg(object_cfg)
        return object_cfg
