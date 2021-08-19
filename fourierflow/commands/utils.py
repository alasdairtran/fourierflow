import os


def get_save_dir(config_path):
    """Determine the path where the experimental results will be saved."""
    parts = config_path.split('/')
    i = parts.index('experiments')
    root_dir = os.path.expandvars('$SM_MODEL_DIR')
    save_dir = os.path.join(root_dir, *parts[i+1:-1])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    return save_dir
