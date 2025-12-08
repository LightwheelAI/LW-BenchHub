#!/bin/bash

FLOOR_PLAN_IMAGE_TAG=$1
floorplanImage="harbor.lightwheel.net/floorplan2usd/scene_gen:${FLOOR_PLAN_IMAGE_TAG}"
floorplanCodeFile="./lw_benchhub/core/scenes/kitchen/kitchen_arena.py"

pip install lightwheel_sdk==0.21

declare -A replaceMap=(
    ["acquire_usd(scene=self.scene_cfg.scene_type, version=self.floorplan_version, exclude_layout_ids=exclude_layouts)"]="acquire_usd(scene=self.scene_cfg.scene_type, version=self.floorplan_version, exclude_layout_ids=exclude_layouts, image = '${floorplanImage}')"
    ["acquire_usd(scene=self.scene_cfg.scene_type, layout_id=layout_id, version=self.floorplan_version, exclude_layout_ids=exclude_layouts)"]="acquire_usd(scene=self.scene_cfg.scene_type, layout_id=layout_id, version=self.floorplan_version, exclude_layout_ids=exclude_layouts, image = '${floorplanImage}')"
    ["acquire_usd(scene=self.scene_cfg.scene_type, layout_id=layout_id, style_id=style_id, version=self.floorplan_version, exclude_layout_ids=exclude_layouts)"]="acquire_usd(scene=self.scene_cfg.scene_type, layout_id=layout_id, style_id=style_id, version=self.floorplan_version, exclude_layout_ids=exclude_layouts, image = '${floorplanImage}')"
)

for key in "${!replaceMap[@]}"; do
    value="${replaceMap[$key]}"
    echo "Processing: $key"
    sed -i "s|${key}|${value}|g" "$floorplanCodeFile"
done
