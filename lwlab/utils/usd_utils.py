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

import os
from pxr import Usd, UsdPhysics, UsdGeom, UsdSkel, PhysxSchema, Gf
from lwlab.core.mdp.helpers.transformations import quaternion_from_euler
import lwlab.utils.math_utils.transform_utils as T
import math


class OpenUsd:
    """USD utility class - encapsulates all USD operation methods"""

    def __init__(self, usd_path=None):
        self.stage = None
        if usd_path:
            self.stage = self.get_stage(usd_path)

    @staticmethod
    def get_stage(usd_path):
        """Get USD Stage"""
        stage = Usd.Stage.Open(usd_path)
        return stage

    @staticmethod
    def get_all_prims(stage, prim=None, prims_list=None):
        """Get all prims"""
        if prims_list is None:
            prims_list = []
        if prim is None:
            prim = stage.GetPseudoRoot()
        for child in prim.GetAllChildren():
            prims_list.append(child)
            OpenUsd.get_all_prims(stage, child, prims_list)
        return prims_list

    @staticmethod
    def classify_prim(prim):
        """Classify prim"""
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return "Articulation"
        elif prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return "RigidBody"
        else:
            return "Normal"

    @staticmethod
    def usd_simplify(stage, usd_path, ref_prim, prim=None):
        """Remove useless rigidbody and collisions in the scene"""
        if prim is None:
            prim = stage.GetPseudoRoot()
        for child in prim.GetAllChildren():
            if any(value.name == child.GetName() for value in ref_prim.values()):
                continue
            if not child.IsValid():
                continue
            if child.HasAPI(UsdPhysics.RigidBodyAPI):
                child.RemoveAPI(UsdPhysics.RigidBodyAPI)
                child.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)
            if child.HasAPI(UsdPhysics.CollisionAPI):
                child.RemoveAPI(UsdPhysics.CollisionAPI)
                child.RemoveAPI(PhysxSchema.PhysxCollisionAPI)
            if child.HasAPI(UsdPhysics.ArticulationRootAPI):
                child.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                child.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
            for attr in ["physxContactReportThreshold", "physxContactReportForceThreshold"]:
                if child.HasAttribute(attr):
                    child.RemoveProperty(attr)
                if child.HasAPI(PhysxSchema.PhysxContactReportAPI):
                    child.RemoveAPI(PhysxSchema.PhysxContactReportAPI)
            if "joint" in str(child.GetName()).lower() or "collisions" in str(child.GetName()).lower():
                stage.RemovePrim(child.GetPath().pathString)
                continue
            OpenUsd.usd_simplify(stage, usd_path, ref_prim, child)
        return stage

    @staticmethod
    def activate_prim(stage, name):
        import omni
        root_prim = stage.GetPseudoRoot()
        all_prims = OpenUsd.get_prim_by_prefix(root_prim, name)
        for prim in all_prims:
            prim.SetActive(True)
        return stage

    @staticmethod
    def deactivate_prim(stage, name):
        import omni
        root_prim = stage.GetPseudoRoot()
        all_prims = OpenUsd.get_prim_by_prefix(root_prim, name)
        for prim in all_prims:
            prim.SetActive(False)
        return stage

    @staticmethod
    def is_articulation_root(prim):
        """Check if prim is articulation root"""
        return prim.HasAPI(UsdPhysics.ArticulationRootAPI)

    @staticmethod
    def is_rigidbody(prim):
        """Check if prim is rigidbody"""
        return prim.HasAPI(UsdPhysics.RigidBodyAPI)

    @staticmethod
    def get_all_joints(stage):
        """Get all joints"""
        joints = []

        def recurse(prim):
            # Check if it's a Joint
            if UsdPhysics.Joint(prim):
                joints.append(prim)
            for child in prim.GetAllChildren():
                recurse(child)
        recurse(stage.GetPseudoRoot())
        return joints

    @staticmethod
    def get_prim_pos_rot_in_world(prim):
        """Get prim position, rotation and scale in world coordinates"""
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            return None, None, None
        matrix = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        try:
            pos, rot, scale = UsdSkel.DecomposeTransform(matrix)
            # pos = matrix.ExtractTranslation()
            # rot = matrix.ExtractRotationQuat()
            pos_list = list(pos)
            quat_list = [rot.GetReal(), rot.GetImaginary()[0], rot.GetImaginary()[1], rot.GetImaginary()[2]]  # wxyz
            return pos_list, quat_list, list(scale)
        except Exception as e:
            print(f"Error decomposing transform for {prim.GetName()}: {e}")
            return None, None, None

    @staticmethod
    def get_articulation_joints(articulation_prim):
        """Get joints of articulation"""
        joints = []

        def recurse(prim):
            # Check if it's a Joint
            if UsdPhysics.Joint(prim):
                joints.append(prim)
            for child in prim.GetChildren():
                recurse(child)
        recurse(articulation_prim)
        return joints

    @staticmethod
    def get_joint_type(joint_prim):
        """Get joint type"""
        joint = UsdPhysics.Joint(joint_prim)
        return joint.GetTypeName()

    @staticmethod
    def get_object_extent(prim, obj_name):
        """Get object extent"""
        obj_prims = OpenUsd.get_prim_by_name(prim, obj_name, only_xform=False)
        if len(obj_prims) == 0:
            raise ValueError(f"Object {obj_name} not found in the scene")
        obj_prim = obj_prims[0]
        size = obj_prim.GetAttribute("extent").Get()
        return size

    @staticmethod
    def scale_size(root_prim, scale_factor=(1.0, 1.0, 1.0)):
        """Scale object size"""
        xformable = UsdGeom.Xformable(OpenUsd.get_prim_by_name(root_prim, "root")[0])
        if not xformable:
            raise ValueError("prim must be xformable")
        xform_ops = xformable.GetOrderedXformOps()
        found_scale = False
        for op in xform_ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                op.Set((scale_factor[0], scale_factor[1], scale_factor[2]))
                found_scale = True
                break
        if not found_scale:
            xformable.AddScaleOp().Set((scale_factor[0], scale_factor[1], scale_factor[2]))

    @staticmethod
    def rotate_upright(root_prim, axis=(0, 1, 0)):
        """Rotate object to be upright"""
        xformable = UsdGeom.Xformable(OpenUsd.get_prim_by_name(root_prim, "root")[0])
        if not xformable:
            raise ValueError("prim must be xformable")
        xformable.Rotate(axis, 0)

    @staticmethod
    def export(stage, path):
        """Export prim to path"""
        stage.Export(path)

    @staticmethod
    def is_fixed_joint(prim):
        """Check if joint is fixed"""
        return prim.GetTypeName() == 'PhysicsFixedJoint'

    @staticmethod
    def is_revolute_joint(prim):
        """Check if joint is revolute"""
        return prim.GetTypeName() == 'PhysicsRevoluteJoint'

    @staticmethod
    def is_prismatic_joint(prim):
        """Check if joint is prismatic"""
        return prim.GetTypeName() == "PhysicsPrismaticJoint"

    @staticmethod
    def get_joint_name_and_qpos(joint_prim):
        """Get joint name and position"""
        joint = UsdPhysics.Joint(joint_prim)
        return joint.GetName(), joint.GetPositionAttr().Get()

    @staticmethod
    def get_all_joints_without_fixed(articulation_prim):
        """Get all non-fixed joints"""
        joints = OpenUsd.get_articulation_joints(articulation_prim)
        return [joint for joint in joints if not OpenUsd.is_fixed_joint(joint)]

    @staticmethod
    def get_prim_by_name(prim, name, only_xform=True):
        """Get prim by name"""
        result = []
        if prim.GetName().lower() == name.lower():
            if not only_xform or prim.GetTypeName() == "Xform":
                result.append(prim)
        for child in prim.GetAllChildren():
            result.extend(OpenUsd.get_prim_by_name(child, name, only_xform))
        return result

    @staticmethod
    def get_prim_by_type(prim, include_types=None, exclude_types=None):
        """Get prim by type"""
        result = []
        if (
            (include_types is None or prim.GetTypeName() in include_types)
            and (exclude_types is None or prim.GetTypeName() not in exclude_types)
        ):
            result.append(prim)
        for child in prim.GetAllChildren():
            result.extend(OpenUsd.get_prim_by_type(child, include_types, exclude_types))
        return result

    @staticmethod
    def get_prim_by_name_and_type(prim, name, type):
        """Get prim by name and type"""
        result = []
        if prim.GetName().lower() == name.lower() and prim.GetTypeName() == type:
            result.append(prim)
        for child in prim.GetAllChildren():
            result.extend(OpenUsd.get_prim_by_name_and_type(child, name, type))
        return result

    @staticmethod
    def get_prim_by_prefix(prim, prefix, only_xform=True):
        """Get prim by prefix"""
        result = []
        if prim.GetName().lower().startswith(prefix.lower()):
            if not only_xform or prim.GetTypeName() == "Xform":
                result.append(prim)
        for child in prim.GetAllChildren():
            result.extend(OpenUsd.get_prim_by_prefix(child, prefix, only_xform))
        return result

    @staticmethod
    def get_prim_by_prefix_and_type(prim, prefix, type):
        """Get prim by prefix and type"""
        result = []
        if prim.GetName().lower().startswith(prefix.lower()) and prim.GetTypeName() == type:
            result.append(prim)
        for child in prim.GetAllChildren():
            result.extend(OpenUsd.get_prim_by_prefix_and_type(child, prefix, type))
        return result

    @staticmethod
    def get_prim_by_suffix(prim, suffix, only_xform=True):
        """Get prim by suffix"""
        result = []
        if prim.GetName().lower().endswith(suffix.lower()):
            if not only_xform or prim.GetTypeName() == "Xform":
                result.append(prim)
        for child in prim.GetAllChildren():
            result.extend(OpenUsd.get_prim_by_suffix(child, suffix, only_xform))
        return result

    @staticmethod
    def get_prim_by_suffix_and_type(prim, suffix, type):
        """Get prim by suffix and type"""
        result = []
        if prim.GetName().lower().endswith(suffix.lower()) and prim.GetTypeName() == type:
            result.append(prim)
        for child in prim.GetAllChildren():
            result.extend(OpenUsd.get_prim_by_suffix_and_type(child, suffix, type))
        return result

    @staticmethod
    def has_contact_reporter(prim):
        """Check if prim has contact reporter"""
        return prim.HasAPI(PhysxSchema.PhysxContactReportAPI)

    @staticmethod
    def get_child_commonprefix_name(prim):
        """Get common prefix name of child elements"""
        child_rigidbody_prims = []
        for child in prim.GetChildren():
            if OpenUsd.is_rigidbody(child):
                child_rigidbody_prims.append(child)
        names = [prim.GetName() for prim in child_rigidbody_prims]

        if len(names) == 1:
            return names[0]

        return os.path.commonprefix(names)

    @staticmethod
    def get_all_child_commonprefix_name(prim):
        """Get common prefix name of child elements"""
        child_rigidbody_prims = []
        for child in prim.GetAllChildren():
            if OpenUsd.is_rigidbody(child):
                child_rigidbody_prims.append(child)
        names = [prim.GetName() for prim in child_rigidbody_prims]

        if len(names) == 1:
            return names[0]

        return os.path.commonprefix(names)

    @staticmethod
    def get_child_xform_infos(prim):
        """Get child xform infos"""
        infos = []
        for child in prim.GetChildren():
            if child.GetTypeName() == "Xform":
                info = dict(
                    name=child.GetName(),
                    type=child.GetAttribute("type").Get(),
                    prim=child,
                )
                infos.append(info)
        return infos

    @staticmethod
    def get_all_child_xform_infos(prim):
        """Get child xform infos"""
        infos = []
        for child in prim.GetAllChildren():
            if child.GetTypeName() == "Xform":
                info = dict(
                    name=child.GetName(),
                    type=child.GetAttribute("type").Get(),
                    prim=child,
                )
                infos.append(info)
        return infos

    @staticmethod
    def get_prim_aabb_bounding_box(prim):
        """Get prim aabb bounding box"""
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        return bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()

    @staticmethod
    def get_fixture_placements(root_prim, fixture_cfgs, fixtures):
        valid_fixture_names = []
        counter_fixture_names = []
        fixture_placements = {}
        counter_fixture_placements = {}
        for fxr_cfg in fixture_cfgs:
            if "ProcGenFixture" not in [cls.__name__ for cls in fxr_cfg["model"].__class__.mro()]:
                valid_fixture_names.append(fxr_cfg["name"])
            elif fxr_cfg['model'].__class__.__name__ == "Counter":
                counter_fixture_names.append(fxr_cfg["name"])
        for name in valid_fixture_names:
            prim = OpenUsd.get_prim_by_name(root_prim, name)
            if len(prim) == 0:
                raise ValueError(f"Fixture {name} not found in the scene")
            prim = prim[0]
            fixture_pos = tuple(prim.GetAttribute("xformOp:translate").Get())
            fixture_rot = prim.GetAttribute("xformOp:rotateXYZ").Get()
            fixture_quat = T.mat2quat(T.euler2mat([math.radians(fixture_rot[0]), math.radians(fixture_rot[1]), math.radians(fixture_rot[2])]))
            fixture_placements[name] = (fixture_pos, fixture_quat, fixtures[name])

        for name in counter_fixture_names:
            prim = OpenUsd.get_prim_by_name(root_prim, name)
            if len(prim) == 0:
                raise ValueError(f"Fixture {name} not found in the scene")
            prim = prim[0]
            fixture_pos = tuple(prim.GetAttribute("xformOp:translate").Get())
            fixture_rot = prim.GetAttribute("xformOp:rotateXYZ").Get()
            fixture_quat = T.mat2quat(T.euler2mat([math.radians(fixture_rot[0]), math.radians(fixture_rot[1]), math.radians(fixture_rot[2])]))
            counter_fixture_placements[name] = (fixture_pos, fixture_quat, fixtures[name])

        return fixture_placements, counter_fixture_placements


class OpenUsdWrapper:
    def __init__(self, usd_path):
        self._usd = OpenUsd(usd_path)

    def __getattr__(self, name):
        return getattr(self._usd, name)

    @property
    def stage(self):
        return self._usd.stage

    @property
    def root_prim(self):
        return self.stage.GetPseudoRoot()

    def get_all_prims(self, prim=None, prims_list=None):
        return self._usd.get_all_prims(self.stage, prim, prims_list)

    def usd_simplify(self, usd_path, ref_prim, prim=None):
        return self._usd.usd_simplify(self.stage, usd_path, ref_prim, prim)

    def activate_prim(self, name):
        return self._usd.activate_prim(self.stage, name)

    def deactivate_prim(self, name):
        return self._usd.deactivate_prim(self.stage, name)

    def get_all_joints(self):
        return self._usd.get_all_joints(self.stage)

    def get_prim_pos_rot_in_world(self, prim):
        return self._usd.get_prim_pos_rot_in_world(prim)

    def get_object_extent(self, obj_name, prim=None):
        if prim is None:
            prim = self.root_prim
        return self._usd.get_object_extent(prim, obj_name)

    def scale_size(self, scale_factor=(1.0, 1.0, 1.0)):
        return self._usd.scale_size(self.root_prim, scale_factor)

    def rotate_upright(self, axis=(0, 1, 0)):
        return self._usd.rotate_upright(self.root_prim, axis)

    def export(self, path):
        return self._usd.export(self.stage, path)

    def get_articulation_joints(self, articulation_prim=None):
        if articulation_prim is None:
            articulation_prim = self.root_prim
        return self._usd.get_articulation_joints(articulation_prim)

    def get_all_joints_without_fixed(self, articulation_prim=None):
        if articulation_prim is None:
            articulation_prim = self.root_prim
        return self._usd.get_all_joints_without_fixed(articulation_prim)

    def get_prim_by_name(self, name, only_xform=True, prim=None):
        if prim is None:
            prim = self.root_prim
        return self._usd.get_prim_by_name(prim, name, only_xform)

    def get_prim_by_name_and_type(self, name, type, prim=None):
        if prim is None:
            prim = self.root_prim
        return self._usd.get_prim_by_name_and_type(self.root_prim, name, type)

    def get_prim_by_prefix(self, prefix, only_xform=True, prim=None):
        if prim is None:
            prim = self.root_prim
        return self._usd.get_prim_by_prefix(prim, prefix, only_xform)

    def get_prim_by_prefix_and_type(self, prefix, type, prim=None):
        if prim is None:
            prim = self.root_prim
        return self._usd.get_prim_by_prefix_and_type(prim, prefix, type)

    def get_prim_by_suffix(self, suffix, only_xform=True, prim=None):
        if prim is None:
            prim = self.root_prim
        return self._usd.get_prim_by_suffix(prim, suffix, only_xform)

    def get_prim_by_suffix_and_type(self, suffix, type, prim=None):
        if prim is None:
            prim = self.root_prim
        return self._usd.get_prim_by_suffix_and_type(prim, suffix, type)

    def get_all_child_commonprefix_name(self, prim=None):
        if prim is None:
            prim = self.root_prim
        return self._usd.get_all_child_commonprefix_name(prim)

    def get_all_child_xform_names(self, prim=None):
        if prim is None:
            prim = self.root_prim
        return self._usd.get_all_child_xform_names(prim)

    def get_fixture_placements(self, fixture_cfgs, fixtures, root_prim=None):
        if root_prim is None:
            root_prim = self.root_prim
        return self._usd.get_fixture_placements(root_prim, fixture_cfgs, fixtures)
