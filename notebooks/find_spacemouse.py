#!/usr/bin/env python
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

import hid


def main():
    for device in hid.enumerate():
        if (
            device["product_string"] == "SpaceMouse Compact"
            or device["product_string"] == "SpaceMouse Wireless"
            or device["product_string"] == "SpaceNavigator for Notebooks"
        ):
            print(f'Found SpaceMouse at: {device["path"]}')
            break
    else:
        print("No SpaceMouse found")


if __name__ == "__main__":
    main()
