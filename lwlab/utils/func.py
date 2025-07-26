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

from contextlib import contextmanager
import cProfile
import pstats
import datetime
from pathlib import Path
import os


@contextmanager
def trace_profile(filename="code", sort="cumtime", limit=50):
    if os.environ.get("TRACE_PROFILE") == "1":
        filename = Path(f"prof/{filename}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.prof")
        filename.parent.mkdir(exist_ok=True, parents=True)
        pr = cProfile.Profile()
        pr.enable()
        try:
            yield
        finally:
            pr.disable()
            pr.dump_stats(str(filename))
            ps = pstats.Stats(pr)
            ps.sort_stats(sort).print_stats(limit)
    else:
        yield
