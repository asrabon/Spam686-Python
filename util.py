
def njit_if_numba(dec, condition):
    """
    Use njit decorator for performance benefits if numba backend is available
    """
    def decorator(func):
        if not condition:
            # Return the function unchanged, not decorated.
            return func
        return dec(func, fastmath=True, parallel=True)
    return decorator