import socket
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import torch

from torch import multiprocessing as tmp
import typing
from .proxy import RemoteEnv


def wait_port_listen(address: typing.Tuple[str, int], timeout: float = 20.0):
    host, port = address
    deadline = time.time() + timeout

    while True:
        try:
            with socket.create_connection((host, port), timeout=1):
                # Connected successfully => port is listening
                return
        except (ConnectionRefusedError, TimeoutError, OSError):
            # Not listening yet
            pass

        if time.time() >= deadline:
            raise TimeoutError(f"Port {host}:{port} did not start listening within {timeout} seconds.")

        time.sleep(0.1)


class EnvWorker(tmp.Process):
    def __init__(self, address: typing.Tuple[str, int], device: str = "cuda:0"):
        super().__init__(daemon=True)
        self.address = address
        self.device = device
        self.client = None
        self._thread_pool = ThreadPoolExecutor(max_workers=1)

    def _call_in_local_pool_async(self, func: typing.Callable, *args, **kwargs):
        return self._thread_pool.submit(func, *args, **kwargs)

    _get_client_future = None

    def start(self):
        res = super().start()
        self._get_client_future = self._call_in_local_pool_async(self._get_client)
        return res

    def wait_ready(self):
        if self._get_client_future is None:
            raise ValueError("Worker not started")
        return self._get_client_future.result()

    def shutdown(self):
        self._thread_pool.shutdown(wait=True, cancel_futures=True)
        try:
            start_time = time.time()
            self.join(timeout=1)
            print(f"workers {self.device} joined in {time.time() - start_time}s")
        except TimeoutError:
            print(f"workers {self.device} timed out, kill it")
            self.kill()
        self.kill()
        try:
            self.close()
        except Exception as e:
            print(f"Error closing worker {self.device}: {e}")

    def run(self):
        # this function is run inside the worker process
        import sys
        # clean the args from parent process
        sys.argv = [sys.argv[0]]
        from lwlab.distributed.ipc import IpcDistributedEnvWrapper
        from lwlab.scripts.env_server import make_env
        from functools import partial

        def asd(t):
            print("asd", t.device)
            import torch
            return torch.rand_like(t)
        with IpcDistributedEnvWrapper(
            env_initializer=partial(
                make_env,
                args_override=dict(device=self.device)
            ),
            address=self.address,
            device=self.device
        ) as env:
            env.asd = asd
            env.serve()

    def _get_client(self):
        wait_port_listen(self.address)
        self.client = RemoteEnv.make(address=self.address)

    def submit(self, func_name: str, args, kwargs):
        def _call_func():
            return getattr(self.client, func_name)(*args, **kwargs)
        return self._thread_pool.submit(_call_func)

    def __getattr__(self, key):
        if key in {"__iter__"}:
            raise AttributeError(f"Worker does not have attribute {key}")
        print(f"worker __getattr__: {key}")
        if self.client is None:
            raise ValueError("Client not connected")

        def getattr_func():
            return getattr(self.client, key)
        return self._thread_pool.submit(getattr_func).result()


class EnvRouter:
    _passthrough_attach: bool = True
    _result_idx_not_merge: set = {}
    _method_kwargs_to_split: typing.Dict[str, typing.List[str]] = {
        "step": ["action"],
        "asd": ["t"],
        "reset_to": ["state"],
    }
    workers = ()

    def __init__(
        self,
        worker_count: int,
        address: typing.Tuple[str, int] = ('0.0.0.0', 5000),
        device: str = "cuda:0",
        app_launcher_args: dict = None
    ):
        self.host, self.port = address
        self.port = int(self.port)
        self.worker_count = worker_count
        self.device = device
        self.app_launcher_args = app_launcher_args

        self._start_workers()

    def __getattr__(self, key):  # DONE
        """
        If the attribute is callable, return a lambda function that calls the attribute and merges the result from all workers.
        Otherwise, return the attribute from the first client.
        """
        if len(self.workers) == 0:
            raise AttributeError("No workers started")
        obj = getattr(self.workers[0], key)
        if callable(obj):
            obj = lambda *args, **kwargs: self._merge_result_from_workers(self._call_in_parallel(key, args, kwargs), key)  # noqa
            setattr(self, key, obj)
        return obj

    def _start_workers(self):  # done
        # tmp.set_start_method("spawn")
        self.workers: typing.List[EnvWorker] = []
        for i in range(self.worker_count):
            worker = EnvWorker(address=(self.host, self.port + i), device=f"cuda:{i}")
            print(f"Starting worker on {f'cuda:{i}'}")
            worker.start()
            self.workers.append(worker)
        for worker in self.workers:
            worker.wait_ready()
        print("All workers ready")

    def _call_in_parallel(self, func_name: str, args=[], kwargs={}, split_kwargs: typing.List[dict] = None, sequential: bool = False):  # done
        # for simplicity, only split args from kwargs
        split_kwargs = split_kwargs or [{} for _ in range(self.worker_count)]
        if func_name in self._method_kwargs_to_split:
            for k in self._method_kwargs_to_split[func_name]:
                if k not in kwargs:
                    continue
                v = kwargs.pop(k)
                split_v = self._split_tensor_to_workers(v)
                for i, v_ in enumerate(split_v):
                    split_kwargs[i][k] = v_
        if sequential:
            return [
                worker.submit(func_name, args, {**kwargs, **split_kwargs[i]}).result()
                for i, worker in enumerate(self.workers)
            ]
        else:
            futures = [
                worker.submit(func_name, args, {**kwargs, **split_kwargs[i]})
                for i, worker in enumerate(self.workers)
            ]
            # Preserve worker order: iterate through futures in submission order, not completion order
            return [future.result() for future in futures]

    def _split_tensor_to_workers(self, tensor: torch.Tensor):  # done
        return [
            tensor_.to(worker.device, copy=True)
            for worker, tensor_ in
            zip(self.workers, tensor.chunk(self.worker_count))
        ]

    def _merge_tensors_from_workers(self, tensors: typing.List[torch.Tensor]):  # done
        return torch.cat([tensor.to(self.device) for tensor in tensors], dim=0)

    def _merge_scalars_from_workers(self, scalars: typing.List[float]):  # done
        if isinstance(scalars[0], torch.Tensor):
            return torch.mean(torch.stack([scalar.to(self.device) for scalar in scalars]))
        else:
            return sum(scalars) / len(scalars)

    def _merge_result_from_workers(self, results: list, idx: str):  # done
        """Merge the results from all workers into a single result.
        If the result is a list, tuple, or dict, merge the results from all workers into a single result.
        If the result is a tensor, merge the results from all workers into a single tensor.
        If the result is None, return None.
        If the result is in the _result_idx_not_merge set, return the result from the first worker.
        Otherwise, return the result from first worker.
        """
        result0 = results[0]

        if result0 is None:
            return None
        if idx in self._result_idx_not_merge:
            return result0
        if isinstance(result0, (list, tuple)):
            return [
                self._merge_result_from_workers(
                    [result[i] for result in results],
                    f"{idx}.{i}"
                )
                for i in range(len(result0))
            ]
        elif isinstance(result0, dict):
            return {
                k: self._merge_result_from_workers([result[k] for result in results], f"{idx}.{k}")
                for k in result0.keys()
            }
        elif (isinstance(result0, torch.Tensor) and len(result0.shape) == 0) or isinstance(result0, float):
            return self._merge_scalars_from_workers(results)
        elif isinstance(result0, torch.Tensor):
            return self._merge_tensors_from_workers(results)
        else:
            return result0

    @property
    def num_envs(self):
        return self.workers[0].num_envs * self.worker_count

    def close(self):
        # self._call_in_parallel("close")
        for worker in self.workers:
            worker.shutdown()

    def step(self, action: torch.Tensor):
        return self._merge_result_from_workers(self._call_in_parallel("step", kwargs=dict(action=action)), "step")

    def reset_to(self, state: dict, env_ids, *args, **kwargs):
        kwargs["state"] = state
        kwargs["env_ids"] = env_ids
        if env_ids is None:
            return self._merge_result_from_workers(self._call_in_parallel("reset_to", args, kwargs), "reset_to")
        else:
            raise NotImplementedError("reset_to with env_ids is not implemented yet")

    def attach(self, cfg):
        if cfg.num_envs % self.worker_count != 0:
            raise ValueError(f"Number of environments must be divisible by the number of workers. Got {cfg.num_envs} environments and {self.worker_count} workers.")
        each_kwargs = []
        for worker in self.workers:
            each_cfg = deepcopy(cfg)
            each_cfg.num_envs = cfg.num_envs // self.worker_count
            each_cfg.device = worker.device
            each_launcher_args = dict(
                self.app_launcher_args,
                device=worker.device
            )
            each_kwargs.append(dict(cfg=each_cfg, launcher_args=each_launcher_args))

        self._call_in_parallel(
            "attach",
            split_kwargs=each_kwargs,
            sequential=True
        )

    def detach(self):
        self._call_in_parallel("detach")
