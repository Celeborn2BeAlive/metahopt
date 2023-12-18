def format_time(time: float) -> str:
    """Format time in seconds to human readable string."""
    if time < 1e-3:  # noqa: PLR2004
        return f"{time*1e6:.2f} us"
    elif time < 1:
        return f"{time*1e3:.2f} ms"
    elif time < 60:  # noqa: PLR2004
        return f"{time:.2f} s"
    elif time < 3600:  # noqa: PLR2004
        return f"{time / 60:.2f} m"
    else:
        return f"{time / 3600:.2f} h"
