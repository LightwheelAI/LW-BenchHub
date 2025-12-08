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

import socket
import threading
from typing import TYPE_CHECKING

from .base import BaseDistributedEnv
from .proxy import EnvManager

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class IpcDistributedEnvWrapper(BaseDistributedEnv):

    def __init__(self, env=None, env_initializer=None, address=('', 8000), authkey=b'lightwheel', device="cuda:0"):
        super().__init__(env=env, env_initializer=env_initializer, address=address)
        print(f"IpcDistributedEnvWrapper initialized with address {address} and authkey {authkey}")
        self._manager = self._create_manager(address=address, authkey=authkey)
        self._server = self._manager.get_server()
        self._shutdown_event = threading.Event()
        self._sock = None
        self._device = device

    def serve(self):
        print(f"Waiting for connection on {self._server.listener.address}...")
        print("Press Ctrl+C to stop the server")

        while not self._shutdown_event.is_set():
            try:
                # Set socket to non-blocking mode temporarily to check for shutdown
                self._server.stop_event = threading.Event()
                self._server.listener._listener._socket.settimeout(1.0)  # 1 second timeout
                c = self._server.listener.accept()
                self._sock = socket.fromfd(c._handle, socket.AF_INET, socket.SOCK_STREAM)
                # print(f"Accepted connection from {self._sock.getpeername()}")
                self._server.listener._listener._socket.settimeout(None)  # 1 second timeout
                self._server.handle_request(c)
                self._sock = None
            except socket.timeout:
                # Timeout occurred, check if we should shutdown
                continue
            except OSError as e:
                if self._shutdown_event.is_set():
                    print("Server shutting down...")
                    break
                else:
                    raise e

        print(f"IPC Server on {self._server.listener.address} stopped.")

    def _get_connection_sock(self, c):
        import socket
        return socket.fromfd(c._handle, socket.AF_INET, socket.SOCK_STREAM)

    def signal_handler(self, signum: int, frame):
        self._shutdown_event.set()
        self.close_connection()
        return super().signal_handler(signum, frame)

    def _create_manager(self, address, authkey):
        # server
        mgr = EnvManager(address=address, authkey=authkey)
        mgr.register_for_server(self)
        return mgr

    def start_connection(self):
        super().start_connection()
        from torch.multiprocessing import Queue  # noqa: F401

        print(f"Starting connection to {self._sock.getpeername()}")

    def close_connection(self):
        self._server.stop_event.set()
        if self._sock is not None:
            print(f"Closing connection to {self._sock.getpeername()}")
        super().close_connection()

    def close(self):
        self._shutdown_event.set()
        super().close()

    # def __setattr__(self, key, value):
    #     if key in ("_env", "_manager", "_server"):
    #         return super().__setattr__(key, value)
    #     return setattr(self._env, key, value)
