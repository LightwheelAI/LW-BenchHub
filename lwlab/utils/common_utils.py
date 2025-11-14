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
