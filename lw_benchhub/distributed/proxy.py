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

import atexit
import traceback
from multiprocessing.managers import BaseManager, RemoteError


class EnvManager(BaseManager):
    def register_for_server(self, env):
        self._env = env

        self.register(
            "EnvService",
            lambda: EnvService(self._env),
            exposed=EXPOSED_METHODS,
        )

    def register_for_client(self):
        self.register(
            "EnvService",
            exposed=EXPOSED_METHODS,
        )


class EnvService:
    """Service for remote access to the environment.

    This class provides a set of RPC methods for accessing the environment from a remote process.
    It is used to expose the environment to a remote process, allowing the remote process to interact with the environment.

    Args:
        env: The environment to expose.

    """

    def __init__(self, env):
        self._env = env

    # ---- Tools: resolve object by path ----
    def _resolve(self, path: str):
        obj = self._env
        if not path:
            return obj
        for seg in path.split('.'):
            obj = getattr(obj, seg)
        return obj

    # ---- RPC primitives ----
    def call(self, path: str, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        target = self._resolve(path)
        # print(f"call {path}")
        try:
            return target(*args, **kwargs)
        except Exception:
            print(f"remote call error on env.{path} with {args=} and {kwargs=}")
            traceback.print_exc()
            raise RemoteError(traceback.format_exc())
        # finally:
        #     print(f"call {path} done")

    def getattr_value(self, path: str):
        return self._resolve(path)

    def setattr_value(self, path: str, value):
        parent_path, _, name = path.rpartition('.')
        parent = self._resolve(parent_path) if parent_path else self._env
        setattr(parent, name, value)
        return True

    def is_callable(self, path: str):
        try:
            obj = self._resolve(path)
        except AttributeError:
            return False
        return callable(obj)

    def repr_at(self, path: str):
        try:
            return repr(self._resolve(path))
        except AttributeError:
            return f"<AttributeError at {path!r}>"

    def instance_check(self, cls, path: str):
        return isinstance(self._resolve(path), cls)


EXPOSED_METHODS = ("call", "getattr_value", "setattr_value", "is_callable", "repr_at", "instance_check")


class _PathView:
    """
    Local view object: holds the same remote proxy (a connection), using path routing.
    - Methods: remote execution (RPC)
    - Ordinary attributes: return by value
    - Special: 'unwrapped' returns another PathView (still reuses the same proxy / connection)
    """
    _svc: EnvService

    def __init__(self, svc_proxy, path: str):
        object.__setattr__(self, "_svc", svc_proxy)
        object.__setattr__(self, "_path", path)

    # --- Attribute access ---
    # @tictoc
    def __getattr__(self, name: str):
        value = self._getattr_value(name)
        object.__setattr__(self, name, value)
        return value

    def _getattr_value(self, name: str):
        full = f"{self._path}.{name}" if self._path else name

        # Convention: methods go RPC; 'unwrapped' returns a new PathView; others return by value
        if name == "unwrapped":
            return _PathView(self._svc, full)

        if self._path == "unwrapped" and name == "cfg":
            return _PathView(self._svc, full)

        if self._svc.is_callable(full):
            # Return a callable object (call via RPC)
            # @tictoc(name)
            def _remote_call(*args, **kwargs):
                try:
                    return self._svc.call(full, args, kwargs)
                except EOFError:
                    raise RuntimeError("ENV server has stopped")
                # except KeyboardInterrupt:
                    # self.close_connection()
            _remote_call.__name__ = name
            return _remote_call

        # Ordinary attributes -> return by value
        return self._svc.getattr_value(full)

    def __setattr__(self, name, value):
        full = f"{self._path}.{name}" if self._path else name
        return self._svc.setattr_value(full, value)

    def __repr__(self):
        return self._svc.repr_at(self._path or "")

    # Allow the current View to be used as a callable object (e.g., some objects are callable)
    def __call__(self, *args, **kwargs):
        return self._svc.call(self._path, args, kwargs)

    # Allow the View to be serialized when there are multiple processes: with the same svc proxy + path
    def __getstate__(self):
        return (self._svc, self._path)

    def __setstate__(self, state):
        object.__setattr__(self, "_svc", state[0])
        object.__setattr__(self, "_path", state[1])

    def __instancecheck__(self, cls):
        return self._svc.instance_check(cls, self._path)


# Semantic sugar: top-level env view


class RemoteEnv(_PathView):
    @classmethod
    def make(cls, address, authkey=b'lightwheel') -> "RemoteEnv":
        mgr = EnvManager(address=address, authkey=authkey)
        mgr.connect()
        mgr.register_for_client()
        svc = mgr.EnvService()  # Only one BaseProxy (single connection/process)
        env = cls(svc, "")
        # reset the connection when new env created.
        # to fix issue when server is restarted.
        try:
            del svc._tls.connection
        except AttributeError:
            pass
        env.start_connection()

        def on_exit():
            # print("at exit close connection")
            try:
                env.close_connection()
            except ConnectionRefusedError as e:
                pass
            except Exception as e:
                print(f"[warning] error closing connection: {e}")
        atexit.register(on_exit)
        return env
