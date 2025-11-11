import os
import uuid
import base64
import threading
import signal
import sys
import traceback
import torch
from io import BytesIO
from typing import Any, Dict, List, Tuple

from PIL import Image
import numpy as np
from flask import Flask, request, jsonify


class RestfulEnvWrapper:
    """
    Wrap an existing Gymnasium/IsaacLab environment as a blocking, single-threaded REST server.

    Contract:
    - Only one request is processed at a time (others will block/wait).
    - All handlers run on the main thread (asserted).
    - Endpoints:
        POST /attach   -> { "session_id": str }
        POST /reset    -> { "obs": [base64_img,...], "info": {...} }
        POST /step     -> { "obs": [...], "reward": float, "done": bool, "truncated": bool, "info": {...} }
        POST /detach   -> { "ok": true }
        POST /shutdown -> { "ok": true, "message": "Server shutting down..." }

    Notes:
    - The wrapper does NOT create the env; you must pass an initialized env.
    - Image extraction tries:
        1) Find (H,W,3) uint8 arrays in obs (up to 3).
        2) Fallback to env.render() if available.
    - Action is expected as a space-separated string: 'dx dy dz rdx rdy rdz o'
    - Server can be stopped with Ctrl+C or by calling the /shutdown endpoint
    """

    def __init__(self, env: Any, host: str = "0.0.0.0", port: int = 8000):
        self.env = env
        self.host = host
        self.port = int(port)
        self._shutdown_requested = False

        # In-memory session store; we support multiple session IDs but share one env instance.
        # Since processing is strictly serialized, concurrent sessions are still safe.
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()  # Strict serialization across all endpoints.

        # Flask app is instance-local to avoid globals.
        self.app = Flask(__name__)
        self._register_routes()

        # Set up signal handlers for graceful shutdown
        self._setup_signal_handlers()

    # -------------------------- Public API --------------------------

    def serve(self):
        """
        Start the blocking, single-threaded server. Do NOT enable debug mode
        (debug reloader creates extra processes/threads).
        """
        # Important: threaded=False and processes=1 to keep it single-threaded & single-process.
        print(f"Starting RESTful environment server on http://{self.host}:{self.port}")
        print("Press Ctrl+C to stop the server...")

        try:
            self.app.run(host=self.host, port=self.port, debug=False, threaded=False, processes=1)
        except KeyboardInterrupt:
            print("\nShutdown requested by user (Ctrl+C)")
            self._shutdown_requested = True
        except Exception as e:
            print(f"Server error: {e}")
            raise
        finally:
            print("Server stopped.")

    # -------------------------- Routes --------------------------

    def _register_routes(self):
        app = self.app

        @app.route("/attach", methods=["POST"])
        def attach():
            with self._locked_mainthread():
                data = request.get_json(force=True, silent=True) or {}
                # env_id/env_config are accepted for compatibility but not used to create env here.
                env_id = data.get("env_id")
                env_config = data.get("env_config") or {}

                sid = str(uuid.uuid4())
                self._sessions[sid] = {"env_id": env_id, "env_config": env_config}
                return jsonify({"session_id": sid})

        @app.route("/task_info", methods=["GET"])
        def task_info():
            with self._locked_mainthread():
                try:
                    lang = ""
                    if hasattr(self.env, "cfg") and hasattr(self.env.cfg, "get_ep_meta"):
                        meta = self.env.cfg.get_ep_meta()
                        lang = meta.get("lang", "")
                    return jsonify({"lang": lang})
                except Exception as e:
                    return self._error(f"failed to get task info: {e}", 500)

        @app.route("/reset", methods=["POST"])
        def reset():
            with self._locked_mainthread():
                data = request.get_json(force=True, silent=True) or {}
                sid = data.get("session_id")
                if not self._valid_session(sid):
                    return self._error("invalid session_id", 404)

                try:
                    res = self.env.reset()
                    if isinstance(res, tuple) and len(res) == 2:
                        obs, info = res
                    else:
                        obs, info = res, {}
                    images_b64 = self._extract_images(obs)
                    lang = ""
                    if hasattr(self.env, "cfg") and hasattr(self.env.cfg, "get_ep_meta"):
                        meta = self.env.cfg.get_ep_meta()
                        lang = meta.get("lang", "")
                    return jsonify({
                        "obs": images_b64,
                        # "obs_tensor": self._tensor_to_jsonable(obs),
                        "info": {"task_str": 'Task:' + lang}
                    })
                except Exception as e:
                    return self._error(f"reset failed: {e}", 500)

        @app.route("/step", methods=["POST"])
        def step():
            with self._locked_mainthread():
                data = request.get_json(force=True, silent=True) or {}
                sid = data.get("session_id")
                action_str = data.get("action")

                if not self._valid_session(sid):
                    return self._error("invalid session_id", 404)
                if not isinstance(action_str, str):
                    return self._error("action must be a space-separated string", 400)

                try:
                    action = self._parse_action_string(action_str)
                except Exception:
                    return self._error("failed to parse action string into float list", 400)

                try:
                    # Gymnasium step API: (obs, reward, terminated, truncated, info)
                    if isinstance(action, np.ndarray):
                        action = torch.from_numpy(action).float()
                    if action.ndim == 1:
                        action = action.unsqueeze(0)
                    for _ in range(10):
                        step_out = self.env.step(action)
                    if not (isinstance(step_out, tuple) and len(step_out) >= 5):
                        return self._error("env.step must return (obs, reward, terminated, truncated, info)", 500)

                    obs, reward, terminated, truncated, info = step_out[:5]
                    images_b64 = self._extract_images(obs)
                    return jsonify({
                        "obs": images_b64,
                        # "obs_tensor": self._tensor_to_jsonable(obs),
                        "reward": reward.detach().cpu().numpy().tolist() if torch.is_tensor(reward) else reward,
                        "done": terminated.detach().cpu().numpy().tolist() if torch.is_tensor(terminated) else terminated,
                        "truncated": truncated.detach().cpu().numpy().tolist() if torch.is_tensor(truncated) else truncated,
                        "info": self._tensor_to_jsonable(info if isinstance(info, dict) else {})

                    })
                except Exception as e:
                    return self._error(f"step failed: {e}", 500)

        @app.route("/detach", methods=["POST"])
        def detach():
            with self._locked_mainthread():
                data = request.get_json(force=True, silent=True) or {}
                sid = data.get("session_id")
                if not self._valid_session(sid):
                    return self._error("invalid session_id", 404)

                # We do NOT close the shared env here; only drop the session.
                self._sessions.pop(sid, None)
                return jsonify({"ok": True})

        @app.route("/shutdown", methods=["POST"])
        def shutdown():
            with self._locked_mainthread():
                print("Shutdown requested via API")
                self._shutdown_requested = True
                # Use a separate thread to shutdown Flask after response is sent

                def shutdown_flask():
                    import time
                    time.sleep(0.1)  # Give time for response to be sent
                    func = request.environ.get('werkzeug.server.shutdown')
                    if func is None:
                        raise RuntimeError('Not running with the Werkzeug Server')
                    func()
                threading.Thread(target=shutdown_flask, daemon=True).start()
                return jsonify({"ok": True, "message": "Server shutting down..."})

    # -------------------------- Helpers --------------------------

    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown on Ctrl+C."""
        def signal_handler(signum, frame):
            print(f"\nReceived signal {signum}, initiating graceful shutdown...")
            self._shutdown_requested = True
            # Force exit if graceful shutdown doesn't work
            sys.exit(0)

        # Handle SIGINT (Ctrl+C) and SIGTERM
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _valid_session(self, sid: str) -> bool:
        return isinstance(sid, str) and sid in self._sessions

    def _error(self, msg: str, code: int = 400):
        if code == 500:
            traceback.print_exc()
        return jsonify({"error": msg}), code

    def _assert_main_thread(self):
        # Ensure all handlers execute in the main thread (required by Isaac/Omni in many cases).
        assert threading.current_thread().name == "MainThread", \
            f"Handler must run in MainThread, got {threading.current_thread().name}"

    def _locked_mainthread(self):
        """
        Context manager that:
        1) Acquires the global lock to enforce strict serialization.
        2) Asserts we are running on the main thread.
        """
        class _Guard:
            def __init__(self, outer: "RestfulEnvWrapper"):
                self.outer = outer

            def __enter__(self):
                self.outer._lock.acquire()
                self.outer._assert_main_thread()

            def __exit__(self, exc_type, exc, tb):
                self.outer._lock.release()
        return _Guard(self)

    # ---- Image extraction ----

    def _np_rgb_to_base64(self, img: np.ndarray, fmt: str = "PNG", include_media_type: bool = False) -> str:
        assert img.ndim == 3 and img.shape[2] == 3, "Expect (H, W, 3) RGB array"
        pil = Image.fromarray(img.astype(np.uint8), mode="RGB")
        buf = BytesIO()
        pil.save(buf, format=fmt)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return (f"data:image/{fmt.lower()};base64," + b64) if include_media_type else b64

    def _collect_rgb_from_any(self, x: Any, found: List[np.ndarray], max_images: int):
        if len(found) >= max_images:
            return
        if isinstance(x, torch.Tensor) and x.ndim == 3 and x.shape[2] == 3:
            found.append(x.cpu().numpy())
        if isinstance(x, torch.Tensor) and x.ndim == 4 and x.shape[3] == 3:
            for i in range(x.shape[0]):
                found.append(x[i].cpu().numpy())
        elif isinstance(x, dict):
            for v in x.values():
                if len(found) >= max_images:
                    break
                self._collect_rgb_from_any(v, found, max_images)
        elif isinstance(x, (list, tuple)):
            for v in x:
                if len(found) >= max_images:
                    break
                self._collect_rgb_from_any(v, found, max_images)

    def _extract_images(self, obs: Any, max_images: int = 30000) -> List[str]:
        """
        Try to extract up to N RGB frames (H,W,3) from obs; fallback to env.render().
        Return base64-encoded PNG strings.
        """
        found: List[np.ndarray] = []
        self._collect_rgb_from_any(obs, found, max_images)
        if found:
            return [self._np_rgb_to_base64(img) for img in found[:max_images]]

        # Fallback to env.render() if available.
        try:
            arr = self.env.render()
            if isinstance(arr, np.ndarray) and arr.ndim == 4 and arr.shape[3] == 3:
                return [self._np_rgb_to_base64(arr)]
        except Exception:
            pass

        # No images available; return empty list (client should handle).
        return []

    # ---- Action parsing ----

    @staticmethod
    def _parse_action_string(action_str: str) -> np.ndarray:
        """
        Parse a space-separated float string into np.ndarray (N,).
        Example: 'dx dy dz rdx rdy rdz o' -> shape (7,)
        """
        parts = [float(x) for x in action_str.strip().split()]
        return np.asarray(parts, dtype=np.float32)

    # ---- Tensor Translation ----

    def _tensor_to_jsonable(self, obs: Any):
        import torch
        import numpy as np

        if isinstance(obs, (torch.Tensor, torch.nn.Parameter)):
            return obs.detach().cpu().numpy().tolist()
        elif isinstance(obs, np.ndarray):
            return obs.tolist()
        elif isinstance(obs, dict):
            return {str(k): self._tensor_to_jsonable(v) for k, v in obs.items()}
        elif isinstance(obs, (list, tuple, set)):
            return [self._tensor_to_jsonable(v) for v in obs]
        elif hasattr(obs, "to") and hasattr(obs, "cpu"):  # IsaacLab TensorDict fallback
            try:
                return self._tensor_to_jsonable(obs.cpu())
            except Exception:
                return str(obs)
        elif isinstance(obs, (float, int, str, bool)) or obs is None:
            return obs
        else:
            # Fallback: convert to string
            return str(obs)
