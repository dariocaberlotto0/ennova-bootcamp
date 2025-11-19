import time

# Decorator to time function execution
def timed(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"Execution: {time.time() - start:.2f}s")
        return result
    return wrapper

def timed_async(func):
    async def wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        print(f"Execution: {time.time() - start:.2f}s")
        return result
    return wrapper