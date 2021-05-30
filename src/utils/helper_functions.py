from datetime import datetime

def get_timestamp(format_str="%Y%m%d_%H-%M-%S"):
    """Helper function that generates a timestamp string in the default format "%Y%M%d_%H-%M-%S".

    Args:
        format_str (str, optional): Can be given to get timestamp in certain format. Defaults to "%Y%M%d_%H-%M-%S".

    Returns:
        [str]: String containing the timestamp in the given format.
    """
    now = datetime.now()
    current_timestamp = now.strftime(format_str)

    return current_timestamp
