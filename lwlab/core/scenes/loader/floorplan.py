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

import grpc
import json
from . import bundle_pb2 as bundle_pb2
from . import bundle_pb2_grpc as bundle_pb2_grpc
from pathlib import Path
import zipfile
from tqdm import tqdm
import threading
import shutil

CACHE_PATH = Path("~/.cache/lwlab/floorplan/").expanduser()
CACHE_PATH.mkdir(parents=True, exist_ok=True)


class FloorplanLoader:
    def __init__(self, host):
        self.host = host
        self.lock = threading.Lock()
        self._args = None
        self.check_version()

    def acquire_usd(self, layout_id: int, style_id: int, scene: str = None):
        if self._args is not None:
            raise RuntimeError("Downloading floorplan is already in progress")
        with self.lock:
            self._t = threading.Thread(target=self.get_usd, args=(layout_id, style_id, scene))
            self._t.start()
        return self

    def wait_for_result(self):
        """wait for the floorplan to be downloaded and return the path"""
        with self.lock:
            return self._usd_file_path

    def get_usd(self, layout_id: int, style_id: int, scene: str = None):
        with self.lock:
            self._args = (layout_id, style_id, scene)
            self._usd_file_path = self._get_usd()
            self._args = None
            return self._usd_file_path

    def check_version(self):
        version = self._get_version()
        if self._usd_cache_version_path().exists():
            with open(self._usd_cache_version_path(), "r") as f:
                cached_version = f.read()
            if cached_version == version:
                return
            self._clear_usd_cache()
            self._update_usd_cache_version(version)
        else:
            self._clear_usd_cache()
            self._update_usd_cache_version(version)

    def _get_usd(self):
        """
        Make a Get RPC call to retrieve a bundle stream.

        Args:
            layout_id (int): The layout ID
            style_id (int): The style ID
            scene (str, optional): The scene identifier

        Returns:
            iterator: Stream of GetBundleReply messages
        """
        layout_id, style_id, scene = self._args
        cache_dir_path = self._usd_cache_dir_path(dict(layout_id=layout_id, style_id=style_id))
        # cache_dir_path.mkdir(parents=True, exist_ok=True)
        package_file_path = cache_dir_path.with_suffix(".zip")
        usd_file_path = cache_dir_path / "scene.usda"
        if usd_file_path.exists():
            return usd_file_path

        with grpc.insecure_channel(self.host) as channel:
            stub = bundle_pb2_grpc.BundleStub(channel)
            request = bundle_pb2.GetBundleRequest(layout_id=layout_id, style_id=style_id, scene=scene if scene else None)
            try:
                with open(package_file_path, "wb") as f:
                    total_size = 0
                    for reply in tqdm(stub.Get(request), desc="Downloading Floorplan Package"):
                        f.write(reply.data)
                        total_size += len(reply.data)
                    print(f"dowloaded {total_size/1024/1024:.2f}MB")
            except grpc.RpcError as e:
                raise e
        # decompress the package.zip to the cache_dir_path
        with zipfile.ZipFile(package_file_path, "r") as zip_ref:
            zip_ref.extractall(CACHE_PATH)
            package_file_path.unlink()
        return usd_file_path

    def _usd_cache_dir_path(self, cache_key_args: dict):
        return CACHE_PATH / f"robocasakitchen-{cache_key_args['layout_id']}-{cache_key_args['style_id']}"

    def _clear_usd_cache(self):
        print(f"clear USD cache at {CACHE_PATH}")
        if not CACHE_PATH.exists():
            return
        for path in CACHE_PATH.glob("*"):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

    def _usd_cache_version_path(self):
        return CACHE_PATH / "version.txt"

    def _update_usd_cache_version(self, version: str):
        print(f"update USD cache version at {self._usd_cache_version_path()}")
        if not CACHE_PATH.exists():
            CACHE_PATH.mkdir(parents=True, exist_ok=True)
        with open(self._usd_cache_version_path(), "w") as f:
            f.write(version)

    def _get_version(self):
        """
        Make a GetVersion RPC call to retrieve the version of the floorplan.

        Returns:
            dict: The version of the floorplan, including image, svn, and worker_version
        """
        with grpc.insecure_channel(self.host) as channel:
            stub = bundle_pb2_grpc.BundleStub(channel)
            request = bundle_pb2.GetVersionRequest()
            reply = stub.GetVersion(request)
            return reply.version


floorplan_loader = FloorplanLoader("usdcache.lightwheel.net:30905")
