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

from functools import wraps


def retry(max_attempts=3):
    """
    Decorator that retries a function up to max_attempts times on exception.

    Args:
        max_attempts (int): Maximum number of retry attempts (default: 3)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        print(f"\nWarning: {func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {e}")
                        print(f"Retrying...")
                    else:
                        print(f"\nError: {func.__name__} failed after {max_attempts} attempts")
                        raise last_exception
            return None
        return wrapper
    return decorator
