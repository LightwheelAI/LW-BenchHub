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

import math
import os
from turtle import st

import numpy as np
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdSkel

import lw_benchhub.utils.math_utils.transform_utils.numpy_impl as T


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
    def usd_simplify(stage, ref_fixture_names, prim=None):
        """Remove useless rigidbody and collisions in the scene"""
        if prim is None:
            prim = stage.GetPseudoRoot()
        for child in prim.GetAllChildren():
            if any(name == child.GetName() for name in ref_fixture_names):
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

            if UsdPhysics.Joint(child):
                stage.RemovePrim(child.GetPath().pathString)
                continue
            OpenUsd.usd_simplify(stage, ref_fixture_names, child)
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
    def has_articulation_root(prim):
        """Recursively check if prim or any of its descendants has ArticulationRootAPI"""
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return True
        for child in prim.GetAllChildren():
            if OpenUsd.has_articulation_root(child):
                return True
        return False

    @staticmethod
    def is_articulation_root(prim):
        """Check if prim is articulation root"""
        return prim.HasAPI(UsdPhysics.ArticulationRootAPI)

    @staticmethod
    def has_rigidbody_api(prim):
        """Recursively check if prim or any of its descendants has RigidBodyAPI"""
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return True
        for child in prim.GetAllChildren():
            if OpenUsd.has_rigidbody_api(child):
                return True
        return False

    @staticmethod
    def is_rigidbody(prim):
        """Check if prim is rigidbody"""
        return prim.HasAPI(UsdPhysics.RigidBodyAPI)

    @staticmethod
    def find_articulation_root(prim):
        """Find articulation root"""
        if OpenUsd.is_articulation_root(prim):
            return prim
        for child in prim.GetAllChildren():
            articulation_root = OpenUsd.find_articulation_root(child)
            if articulation_root:
                return articulation_root
        return None

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
            return None, None
        matrix = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        try:
            pos, rot, scale = UsdSkel.DecomposeTransform(matrix)
            # pos = matrix.ExtractTranslation()
            # rot = matrix.ExtractRotationQuat()
            pos_list = list(pos)
            quat_list = [rot.GetReal(), rot.GetImaginary()[0], rot.GetImaginary()[1], rot.GetImaginary()[2]]  # wxyz
            scale_list = list(scale)
            return pos_list, quat_list, scale_list
        except Exception as e:
            print(f"Error decomposing transform for {prim.GetName()}: {e}")
            return None, None

    @staticmethod
    def get_prim_size(prim):
        bbox = OpenUsd.get_prim_aabb_bounding_box(prim)
        min_point = bbox.GetMin()
        max_point = bbox.GetMax()
        size = max_point - min_point
        return size

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
    def set_contact_force_threshold(stage, root_prim, name, contact_force_threshold=0.0):
        prim = OpenUsd.get_prim_by_name(root_prim, name)[0]
        if not prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
            cr_api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
        else:
            cr_api = PhysxSchema.PhysxContactReportAPI.Get(stage, prim.GetPrimPath())
        cr_api.CreateThresholdAttr().Set(contact_force_threshold)

    @staticmethod
    def set_rgb(root_prim, rgb):
        """Set object rgb"""
        xformable = UsdGeom.Xformable(OpenUsd.get_prim_by_name(root_prim, "root")[0])
        if not xformable:
            raise ValueError("prim must be xformable")

        shader_prims = OpenUsd.get_prim_by_name_and_type(root_prim, "Shader", "Shader")
        if shader_prims.__len__() == 0:
            raise ValueError("prim has no shader")

        for shader_prim in shader_prims:
            shader = UsdShade.Shader(shader_prim)
            if not shader:
                continue
            diffuse_tint_input = shader.GetInput("inputs:diffuse_tint")
            if not (diffuse_tint_input and diffuse_tint_input.GetAttr().IsValid()):
                diffuse_tint_input = shader.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f)
            diffuse_tint_input.Set((rgb[0], rgb[1], rgb[2]))

    @staticmethod
    def convert_asset_instanceable(asset_usd_path, save_as_path=None, create_xforms=True):
        """
            Makes all mesh/geometry prims instanceable (and optionally inserts Xform parents) while preserving materials.
            All operations are performed within the save_as_path file; no additional mesh files are generated.
        """
        import omni.usd

        working_path = save_as_path if save_as_path else asset_usd_path
        root_path = "/"

        if create_xforms:
            omni.usd.get_context().open_stage(asset_usd_path)
            stage = omni.usd.get_context().get_stage()
            prims = [stage.GetPrimAtPath(root_path)]
            edits = Sdf.BatchNamespaceEdit()

            while len(prims) > 0:
                prim = prims.pop(0)
                if prim.GetTypeName() in ["Mesh", "Capsule", "Sphere", "Box"]:
                    new_xform = UsdGeom.Xform.Define(
                        stage, str(prim.GetPath()) + "_xform")
                    edits.Add(Sdf.NamespaceEdit.Reparent(
                        prim.GetPath(), new_xform.GetPath(), 0))
                    continue
                children_prims = prim.GetChildren()
                prims += children_prims

            stage.GetRootLayer().Apply(edits)
            if save_as_path is None:
                omni.usd.get_context().save_stage()
            else:
                omni.usd.get_context().save_as_stage(save_as_path)

        omni.usd.get_context().open_stage(working_path)
        stage = omni.usd.get_context().get_stage()
        prims = [stage.GetPrimAtPath(root_path)]

        while len(prims) > 0:
            prim = prims.pop(0)
            if prim:
                if prim.GetTypeName() in ["Mesh", "Capsule", "Sphere", "Box"]:
                    parent_prim = prim.GetParent()
                    if parent_prim and not parent_prim.IsInstance():
                        parent_prim.SetInstanceable(True)
                        continue
                children_prims = prim.GetChildren()
                prims += children_prims

        if save_as_path is None:
            omni.usd.get_context().save_stage()
        else:
            omni.usd.get_context().save_as_stage(save_as_path)

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
    def get_prim_by_name(prim, name, only_xform=True, case_sensitive=False, only_first=False):
        """Get prim by name"""
        result = []
        if not case_sensitive:
            lower_name = name.lower()
            for child in Usd.PrimRange(prim):
                if child.GetName().lower() == lower_name:
                    if not only_xform or child.GetTypeName() == "Xform":
                        result.append(child)
                        if only_first:
                            break
                # result.extend(OpenUsd.get_prim_by_name(child, name, only_xform))
        else:
            for child in Usd.PrimRange(prim):
                if child.GetName() == name:
                    if not only_xform or child.GetTypeName() == "Xform":
                        result.append(child)
                        if only_first:
                            break

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
    def get_prim_by_subname(prim, subname):
        """Get prim by subname"""
        result = []
        if subname in prim.GetName():
            result.append(prim)
        for child in prim.GetAllChildren():
            result.extend(OpenUsd.get_prim_by_subname(child, subname))
        return result

    @staticmethod
    def get_prim_by_subname_and_type(prim, subname, type):
        """Get prim by subname and type"""
        result = []
        if subname in prim.GetName() and prim.GetTypeName() == type:
            result.append(prim)
        for child in prim.GetAllChildren():
            result.extend(OpenUsd.get_prim_by_subname_and_type(child, subname, type))
        return result

    @staticmethod
    def get_prim_by_types(prim, types):
        """Get prim by types"""
        result = []
        if prim.GetTypeName() in types:
            result.append(prim)
        for child in prim.GetAllChildren():
            result.extend(OpenUsd.get_prim_by_types(child, types))
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
    def get_prim_aabb_bounding_box(prim, use_cache=True):
        """Get prim aabb bounding box without caching"""
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        if not use_cache:
            # Clear any existing cache to ensure fresh computation
            bbox_cache.Clear()
        return bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()

    @staticmethod
    def get_fixture_placements(root_prim, fixture_cfgs):
        valid_fixture_names = []
        fixture_placements = {}
        for fxr_cfg in fixture_cfgs:
            if "ProcGenFixture" not in [cls.__name__ for cls in fxr_cfg["model"].__class__.mro()] and \
               "room" not in fxr_cfg["name"].lower():  # room fixture doesnt have reg_*
                valid_fixture_names.append(fxr_cfg["name"])
        for name in valid_fixture_names:
            prim = OpenUsd.get_prim_by_name(root_prim, name)
            if len(prim) == 0:
                raise ValueError(f"Fixture {name} not found in the scene")
            prim = prim[0]
            fixture_pos = tuple(prim.GetAttribute("xformOp:translate").Get())
            fixture_rot = prim.GetAttribute("xformOp:rotateXYZ").Get()
            fixture_quat = T.mat2quat(
                T.euler2mat([
                    math.radians(fixture_rot[0]),
                    math.radians(fixture_rot[1]),
                    math.radians(fixture_rot[2])
                ])
            )  # xyzw

            from lw_benchhub.utils.place_utils.usd_object import USDObject
            usd_obj = USDObject(
                name=name,
                task_name=name,
                prim=prim,
                obj_path=None,
                object_scale=(1.0, 1.0, 1.0),
                asset_type="fixtures"
            )

            fixture_placements[name] = (fixture_pos, fixture_quat, usd_obj)

        return fixture_placements

    @staticmethod
    def replace_prim(stage, prim, new_prim_path):
        translate_attr = None
        rotateXYZ_attr = None
        scale_attr = None
        need_reorder = False
        path = prim.GetPath()
        old_translate = prim.GetAttribute("xformOp:translate").Get()
        old_rot = prim.GetAttribute("xformOp:rotateXYZ").Get()
        old_bbox = OpenUsd.get_prim_aabb_bounding_box(prim)
        old_bbox_size = np.array([old_bbox.GetSize()[0], old_bbox.GetSize()[1], old_bbox.GetSize()[2]])
        old_bbox_size = abs(T.euler2mat(np.array(old_rot) / 180 * np.pi) @ old_bbox_size)
        prim_type = prim.GetAttribute("type").Get()
        stage.RemovePrim(path)
        new_prim = stage.DefinePrim(path, "Xform")
        new_prim.GetReferences().AddReference(new_prim_path)
        new_bbox = OpenUsd.get_prim_aabb_bounding_box(new_prim)
        new_bbox_size = [new_bbox.GetSize()[0], new_bbox.GetSize()[1], new_bbox.GetSize()[2]]
        relative_scale = [float((old_bbox_size[i]) / new_bbox_size[i]) for i in range(3)]
        xformable = UsdGeom.Xformable(new_prim)
        ops = xformable.GetOrderedXformOps()
        for op in ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                scale_attr = op
                op.Set(Gf.Vec3f(relative_scale[0], relative_scale[1], relative_scale[2]))
            elif op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
                rotateXYZ_attr = op
                op.Set(Gf.Vec3f(old_rot[0], old_rot[1], old_rot[2]))
            elif op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_attr = op
                op.Set(Gf.Vec3f(old_translate[0], old_translate[1], old_translate[2]))
            elif op.GetOpType() in [UsdGeom.XformOp.TypeRotateZYX, UsdGeom.XformOp.TypeOrient]:
                rotateXYZ_attr = xformable.AddRotateXYZOp()
                rotateXYZ_attr.Set(Gf.Vec3f(old_rot[0], old_rot[1], old_rot[2]))
                need_reorder = True
        if need_reorder:
            xformable.SetXformOpOrder([
                translate_attr,
                rotateXYZ_attr,
                scale_attr,
            ])
        type_attr = new_prim.CreateAttribute("type", Sdf.ValueTypeNames.String)
        type_attr.Set(prim_type)
        size_attr = new_prim.CreateAttribute("size", Sdf.ValueTypeNames.String)
        size_attr.Set(f"{old_bbox_size[0]}, {old_bbox_size[1]}, {old_bbox_size[2]}")


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

    def usd_simplify(self, ref_prim_names, prim=None):
        return self._usd.usd_simplify(self.stage, ref_prim_names, prim)

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

    def find_articulation_root(self, prim=None):
        if prim is None:
            prim = self.root_prim
        return self._usd.find_articulation_root(prim)

    def scale_size(self, scale_factor=(1.0, 1.0, 1.0)):
        return self._usd.scale_size(self.root_prim, scale_factor)

    def set_contact_force_threshold(self, name, contact_force_threshold=0.0):
        return self._usd.set_contact_force_threshold(self.stage, self.root_prim, name, contact_force_threshold)

    def set_rgb(self, rgb=(1.0, 1.0, 1.0)):
        return self._usd.set_rgb(self.root_prim, rgb)

    def convert_asset_instanceable(self, save_as_path=None):
        return self._usd.convert_asset_instanceable(self.stage, save_as_path)

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

    def get_prim_by_subname(self, subname, prim=None):
        if prim is None:
            prim = self.root_prim
        return self._usd.get_prim_by_subname(prim, subname)

    def get_prim_by_subname_and_type(self, subname, type, prim=None):
        if prim is None:
            prim = self.root_prim
        return self._usd.get_prim_by_subname_and_type(prim, subname, type)

    def get_prim_by_types(self, types, prim=None):
        if prim is None:
            prim = self.root_prim
        return self._usd.get_prim_by_types(prim, types)

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

    def replace_prim(self, prim, new_prim_path):
        return self._usd.replace_prim(self.stage, prim, new_prim_path)
