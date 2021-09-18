import os
from datetime import datetime
from pathlib import Path


def get_save_dir(config_path):
    """Determine the path where the experimental results will be saved."""
    parts = config_path.split('/')
    i = parts.index('experiments')
    root_dir = os.path.expandvars('$SM_MODEL_DIR')
    save_dir = os.path.join(root_dir, *parts[i+1:-1])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    return save_dir


def get_experiment_id(checkpoint_id, trial, save_dir, resume):
    chkpt_dir = Path(save_dir) / 'checkpoints'
    if resume and not checkpoint_id and chkpt_dir.exists:
        paths = chkpt_dir.glob('*/last.ckpt')
        checkpoint_id = next(paths).parent.name
    now = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    return checkpoint_id or f'trial-{trial}-{now}'
