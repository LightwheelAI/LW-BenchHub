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
from pathlib import Path


def find_folder(folder_name, assets_dir):
    folder_paths = []
    for root, dirs, files in os.walk(str(assets_dir)):
        for dir in dirs:
            if folder_name.lower() in dir.lower() \
                    or folder_name.lower().replace('_', ' ') in dir.lower() \
                    and any(char.isdigit() for char in dir):
                folder_paths.append(str((Path(root) / dir).relative_to(assets_dir) / 'model.xml'))
    return folder_paths


if __name__ == "__main__":
    """
    get the relative path of the model.yaml of the specific asset name
    """
    assets_dir = Path('/home/chaorui.zhang/Fanxiang/lw_benchhub/third_party/robocasa/robocasa/models/assets')
    folder_paths = find_folder("Cup030", assets_dir)
    for obj in folder_paths:
        print(f'/{obj}')
