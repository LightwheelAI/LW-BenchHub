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
from pxr import Usd, UsdPhysics, UsdGeom, UsdSkel, PhysxSchema


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
        for child in prim.GetChildren():
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
            for child in prim.GetChildren():
                recurse(child)
        recurse(stage.GetPseudoRoot())
        return joints

    @staticmethod
    def get_prim_pos_rot_in_world(prim):
        """Get prim position, rotation and scale in world coordinates"""
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            return None, None
        matrix = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

        pos, rot, scale = UsdSkel.DecomposeTransform(matrix)
        # pos = matrix.ExtractTranslation()
        # rot = matrix.ExtractRotationQuat()
        pos_list = list(pos)
        rot_list = [rot.GetReal(), rot.GetImaginary()[0], rot.GetImaginary()[1], rot.GetImaginary()[2]]
        return pos_list, rot_list, list(scale)

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
        for child in prim.GetChildren():
            result.extend(OpenUsd.get_prim_by_name(child, name, only_xform))
        return result

    @staticmethod
    def get_prim_by_name_and_type(prim, name, type):
        """Get prim by name and type"""
        result = []
        if prim.GetName().lower() == name.lower() and prim.GetTypeName() == type:
            result.append(prim)
        for child in prim.GetChildren():
            result.extend(OpenUsd.get_prim_by_name_and_type(child, name, type))
        return result

    @staticmethod
    def get_prim_by_prefix(prim, prefix, only_xform=True):
        """Get prim by prefix"""
        result = []
        if prim.GetName().lower().startswith(prefix.lower()):
            if not only_xform or prim.GetTypeName() == "Xform":
                result.append(prim)
        for child in prim.GetChildren():
            result.extend(OpenUsd.get_prim_by_prefix(child, prefix, only_xform))
        return result

    @staticmethod
    def get_prim_by_prefix_and_type(prim, prefix, type):
        """Get prim by prefix and type"""
        result = []
        if prim.GetName().lower().startswith(prefix.lower()) and prim.GetTypeName() == type:
            result.append(prim)
        for child in prim.GetChildren():
            result.extend(OpenUsd.get_prim_by_prefix_and_type(child, prefix, type))
        return result

    @staticmethod
    def get_prim_by_suffix(prim, suffix, only_xform=True):
        """Get prim by suffix"""
        result = []
        if prim.GetName().lower().endswith(suffix.lower()):
            if not only_xform or prim.GetTypeName() == "Xform":
                result.append(prim)
        for child in prim.GetChildren():
            result.extend(OpenUsd.get_prim_by_suffix(child, suffix, only_xform))
        return result

    @staticmethod
    def get_prim_by_suffix_and_type(prim, suffix, type):
        """Get prim by suffix and type"""
        result = []
        if prim.GetName().lower().endswith(suffix.lower()) and prim.GetTypeName() == type:
            result.append(prim)
        for child in prim.GetChildren():
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
    def get_child_xform_names(prim):
        """Get child xform names"""
        xform_names = []
        for child in prim.GetChildren():
            if child.GetTypeName() == "Xform":
                xform_names.append(child.GetName())
        return xform_names
