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

import abc
import signal
from types import MethodType, FunctionType
from typing import Tuple, Callable, TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def is_property(obj, attr_name):
    cls = type(obj)
    attr = getattr(cls, attr_name, None)
    return isinstance(attr, property)


def generate_env_attrs_meta_info(env):
    meta_info = {}
    attr_names = dir(env)
    for attr_name in attr_names:
        if not attr_name.startswith("_"):
            attr = getattr(env, attr_name)
            if callable(attr):
                meta_info[attr_name] = {
                    "callable": True,
                    "doc": attr.__doc__,
                    "type": f"{type(attr).__module__}.{type(attr).__name__}",
                    "is_method": isinstance(attr, MethodType),
                    "is_function": isinstance(attr, FunctionType),
                    "is_class": isinstance(attr, type),
                }
            else:
                attr_is_property = is_property(env, attr_name)
                attr_type_name = f"{type(attr).__module__}.{type(attr).__name__}"
                if attr is None and attr_is_property:
                    property_func = getattr(env.__class__, attr_name)
                    fget = property_func.fget
                    if "return" in fget.__annotations__:
                        attr_type_name = str(fget.__annotations__["return"])
                attr_is_property = is_property(env, attr_name)

                meta_info[attr_name] = {
                    "callable": False,
                    "type": attr_type_name,
                    "is_property": attr_is_property,
                }
    return meta_info


class BaseDistributedEnv(abc.ABC):
    """Abstract base class for distributed environment wrappers."""
    host: str = None
    port: int = None
    _env_initializer: Optional[Callable] = None
    _env: Optional["ManagerBasedEnv"] = None
    _should_stop: bool = False
    _connected: bool = False
    _passthrough_attach: bool = False

    def __init__(
        self,
        env: Optional["ManagerBasedEnv"],
        env_initializer: Optional[Callable[..., "ManagerBasedEnv"]] = None,
        address: Tuple[str, int] = ('0.0.0.0', 8000)
    ):
        self.host, self.port = address
        self.port = int(self.port)

        if env is not None:
            self._env_initializer = None
            self._env = env
        elif env_initializer is not None:
            self._env_initializer = env_initializer
            self._env = None
        else:
            raise ValueError("Either env or env_initializer must be provided.")
        self._setup_signal_handlers()

    def __getattr__(self, key):
        # print(f"__getattr__: {key}")
        if self._env is None:
            raise AttributeError(f"Environment is not attached, cannot access attribute '{key}'.")
        return getattr(self._env, key)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @abc.abstractmethod
    def serve(self):
        """Start serving the environment."""
        pass

    @abc.abstractmethod
    def close(self):
        """Close the environment and clean up resources."""
        print("Closing environment")
        if self._env is not None:
            self._env.close()
            print(f"{type(self)}._env closed")
            self._env = None

    def start_connection(self):
        self._connected = True

    def close_connection(self):
        self._connected = False

    def attach(self, *args, **kwargs):
        if self._env is not None:
            if self._env._passthrough_attach:
                return self._env.attach(*args, **kwargs)
            raise RuntimeError("Environment is already attached.")
        elif self._env_initializer is None:
            raise RuntimeError("No environment initializer provided.")
        else:
            self._env = self._env_initializer(*args, **kwargs)
        print(f"[INFO-{self.port}]: Attached environment to {self._env}")

    def detach(self):
        if self._env is None:
            raise RuntimeError("Environment is not attached.")
        elif hasattr(self._env, "_passthrough_attach") and self._env._passthrough_attach:
            return self._env.detach()
        elif self._env_initializer is None:
            raise RuntimeError("No environment initializer provided, cannot re-attach.")
        else:
            print("[INFO]: Detaching environment")
            self._env.close()
            print("[INFO]: Environment closed")
            self._env = None
            import omni.usd
            print("[INFO]: new stage")
            omni.usd.get_context().new_stage()
            print("[INFO]: gc")
            import gc
            gc.collect()

    @abc.abstractmethod
    def signal_handler(self, signum: int, frame):
        # get pid
        import os
        pid = os.getpid()
        print(f"\n{pid}:Received signal {signum}, shutting down gracefully...")
        self._should_stop = True

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
