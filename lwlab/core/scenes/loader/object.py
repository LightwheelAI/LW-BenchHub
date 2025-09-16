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

from pathlib import Path
import zipfile
import requests
from .exception import ApiException

CACHE_PATH = Path("~/.cache/lwlab/object/").expanduser()
CACHE_PATH.mkdir(parents=True, exist_ok=True)
from . import login_client


class ObjectLoader:
    """
    Load an object from the floorplan service.

    Args:
        host (str): The host of the API
    """

    def __init__(self, host):
        self.host = host
        self.headers = login_client.get_headers()

    def acquire_object(self, rel_path, file_type: str):
        try:
            return self._acquire_object(rel_path, file_type)
        except ApiException as e:
            if e.authenticated_failed():
                # login and retry
                self.headers = login_client.login(force_login=True)
                return self._acquire_object(rel_path, file_type)
            print(e)
        finally:
            pass

    def _acquire_object(self, rel_path, file_type: str):
        """
        Acquire an object from the floorplan.

        Args:
            levels (list[str]): The levels of the object
            file_type (str): The type of the object, USD, MJCF
        """
        rel_path = rel_path.strip("/")
        levels = rel_path.split("/")
        file_type_to_enum = {"USD": 1, "MJCF": 2}
        if len(levels) > 6 or len(levels) == 0:
            raise ValueError(f"Invalid levels number: {len(levels)}")
        file_type_enum = file_type_to_enum.get(file_type, "")
        if file_type_enum == "":
            raise ValueError(f"Invalid file type: {file_type}")
        payload = {
            "file_type": file_type_enum,
        }
        for i, level in enumerate(levels):
            payload[f"level{i+1}"] = level
        try:
            response = requests.post(
                f"{self.host}/floorplan/v1/levels/get-object",
                json=payload,
                timeout=60,
                headers=self.headers
            )
            if response.status_code != 200:
                raise ApiException(response)
            s3_url = response.json()["fileUrl"]
        except Exception as e:
            raise e

        # download the file to cache
        filename = rel_path.split("/")[-1]
        cache_file_path = CACHE_PATH / (filename + ".zip")
        if not cache_file_path.exists():
            r = requests.get(s3_url, timeout=300)
            if r.status_code != 200:
                raise ApiException(r)
            with open(cache_file_path, "wb") as f:
                f.write(r.content)
        with zipfile.ZipFile(cache_file_path, 'r') as zip_ref:
            zip_ref.extractall(CACHE_PATH)
        cache_file_path.unlink()
        if file_type == "USD":
            return str(CACHE_PATH / filename / (filename + ".usd"))
        elif file_type == "MJCF":
            return str(CACHE_PATH / filename / "model.xml")
