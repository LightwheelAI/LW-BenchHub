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

import cProfile
import datetime
import os
import pstats
import time
from collections import deque, defaultdict
from contextlib import contextmanager
from pathlib import Path

# all active profilers
_active_profilers = []

# current session folder for profiling
_current_session_folder = None


@contextmanager
def trace_profile(filename="code", sort="cumtime", limit=80):
    if os.environ.get("TRACE_PROFILE") == "1":
        global _current_session_folder

        # if first time call, create session folder
        if _current_session_folder is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            _current_session_folder = Path(f"prof/session_{timestamp}")
            _current_session_folder.mkdir(exist_ok=True, parents=True)
            print(f"Created new profiling session: {_current_session_folder}")

        # create prof file in session folder
        prof_file = _current_session_folder / f"{filename}.prof"

        pr = cProfile.Profile()
        pr.enable()

        # add to active profiler list
        profiler_info = {
            'profiler': pr,
            'filename': prof_file,
            'sort': sort,
            'limit': limit
        }
        _active_profilers.append(profiler_info)

        try:
            yield
        finally:
            try:
                pr.disable()
                pr.dump_stats(str(prof_file))
                ps = pstats.Stats(pr)
                ps.sort_stats(sort).print_stats(limit)
                print(f"Profiling data saved to: {prof_file}")

                # remove from active profiler list
                _active_profilers.remove(profiler_info)
            except Exception as e:
                print(f"Error in trace_profile cleanup: {e}")
    else:
        yield


class FrameRateAnalyzer:
    def __init__(self, window_size=60):
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.frame_count = 0
        self.start_time = time.time()

        # stage time statistics
        self.stage_times = defaultdict(list)
        self.stage_names = [
            'total_frame',
            'teleop_advance',
            'env_step',
            'vr_processing',
            'rate_limiter',
            'env_render'
        ]

        # initialize statistics
        for stage in self.stage_names:
            self.stage_times[stage] = []

    def start_frame(self):
        """start a new frame"""
        self.frame_start = time.time()
        self.frame_count += 1
        self.current_stage_start = self.frame_start

    def end_frame(self):
        """end the current frame"""
        frame_time = time.time() - self.frame_start
        self.frame_times.append(frame_time)
        self.stage_times['total_frame'].append(frame_time)

        # calculate average FPS
        if len(self.frame_times) >= 2:
            avg_fps = len(self.frame_times) / sum(self.frame_times)
            print(f"Frame {self.frame_count}: total time {frame_time:.6f}s, average FPS: {avg_fps:.1f}")

    def record_stage(self, stage_name, duration):
        """record the duration of a stage"""
        self.stage_times[stage_name].append(duration)
        print(f"  {stage_name}: {duration:.6f}s")

    def print_results(self):
        """print detailed analysis results"""
        if self.frame_count == 0:
            print("No frames recorded for analysis.")
            return

        print(f"\n=== Frame Rate Analysis ===")
        print(f"Total frames: {self.frame_count}")

        if self.frame_times:
            total_time = sum(self.frame_times)
            avg_frame_time = total_time / len(self.frame_times)
            avg_fps = len(self.frame_times) / total_time
            print(f"Average frame time: {avg_frame_time:.4f}s")
            print(f"Average FPS: {avg_fps:.2f}")

        print(f"\n=== Stage Timing ===")
        for stage_name, times in self.stage_times.items():
            if times:
                avg_time = sum(times) / len(times)
                max_time = max(times)
                min_time = min(times)
                print(f"{stage_name}: avg={avg_time:.4f}s, min={min_time:.4f}s, max={max_time:.4f}s")


# ============================================================================
# Debug Utilities
# ============================================================================

def is_debug_mode() -> bool:
    """check if in debug mode"""
    return os.environ.get("DEBUG_MODE", "false").lower() == "true"


class DebugFrameAnalyzer:
    """frame rate analyzer in debug mode"""

    def __init__(self):
        self.frame_analyzer = None
        if is_debug_mode():
            self.frame_analyzer = FrameRateAnalyzer()

    def start_frame(self):
        """start frame analysis"""
        if self.frame_analyzer is not None:
            self.frame_analyzer.start_frame()

    def record_stage(self, stage_name: str, duration: float):
        """record stage duration"""
        if self.frame_analyzer is not None:
            self.frame_analyzer.record_stage(stage_name, duration)

    def end_frame(self):
        """end frame analysis"""
        if self.frame_analyzer is not None:
            self.frame_analyzer.end_frame()

    def print_results(self):
        """print analysis results"""
        if self.frame_analyzer is not None:
            self.frame_analyzer.print_results()


def debug_print(*args, **kwargs):
    """print only in debug mode"""
    if is_debug_mode():
        print(*args, **kwargs)


DEBUG_FRAME_ANALYZER = DebugFrameAnalyzer()


def tictoc(name):
    def wrapper(func):
        return func

        def wrapper_inner(*args, **kwargs):
            start_time = datetime.now()
            result = func(*args, **kwargs)
            end_time = datetime.now()
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}: {name}{args[1:]} took {(end_time - start_time).total_seconds()*1000:.2f}ms")
            return result
        return wrapper_inner
    if isinstance(name, str):
        return wrapper
    else:
        func = name
        name = func.__name__
        return wrapper(func=func)
