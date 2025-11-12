import threading
import signal
import socket
from .proxy import EnvManager
from typing import Callable, TYPE_CHECKING, Union
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
from .base import BaseDistributedEnv

# for debugging
# import multiprocessing as mp
# mp.util.log_to_stderr()


class IpcDistributedEnvWrapper(BaseDistributedEnv):

    def __init__(self, env, address=('', 8000), authkey=b'lightwheel'):
        super().__init__(env, address=address)
        self._manager = self._create_manager(address=address, authkey=authkey)
        self._server = self._manager.get_server()
        self._shutdown_event = threading.Event()

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
                print(f"Accepted connection from {self._sock.getpeername()}")
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

        print("Server stopped.")

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
        print(f"Closing connection to {self._sock.getpeername()}")
        super().close_connection()

    def close(self):
        self._shutdown_event.set()
        super().close()

    # def __setattr__(self, key, value):
    #     if key in ("_env", "_manager", "_server"):
    #         return super().__setattr__(key, value)
    #     return setattr(self._env, key, value)
