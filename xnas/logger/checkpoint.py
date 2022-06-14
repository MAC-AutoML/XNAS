"""Functions that handle saving and loading of checkpoints"""

import os
import torch
from xnas.core.config import cfg

# Checkpoints directory name
_DIR_NAME = "checkpoints"
# Common prefix for checkpoint file names
_NAME_PREFIX = "model_epoch_"


def get_checkpoint_dir(out_dir=None):
    """Retrieves the location for storing checkpoints."""
    if out_dir is None:
        return os.path.join(cfg.OUT_DIR, _DIR_NAME)
    else:
        return os.path.join(out_dir, _DIR_NAME)


def get_checkpoint_name(epoch, checkpoint_dir=None, best=False):
    """Retrieves the path to a checkpoint file."""
    name = "{}{:04d}.pyth".format(_NAME_PREFIX, epoch)
    name = "best_" + name if best else name
    if checkpoint_dir is None:
        return os.path.join(get_checkpoint_dir(), name)
    else:
        return os.path.join(checkpoint_dir, name)


def get_last_checkpoint(checkpoint_dir=None, best=False):
    """Retrieves the most recent checkpoint (highest epoch number)."""
    if checkpoint_dir is None:
        checkpoint_dir = get_checkpoint_dir()
    # Checkpoint file names are in lexicographic order
    filename = "best_" + _NAME_PREFIX if best else _NAME_PREFIX
    checkpoints = [f for f in os.listdir(checkpoint_dir) if filename in f]
    last_checkpoint_name = sorted(checkpoints)[-1]
    return os.path.join(checkpoint_dir, last_checkpoint_name)


def has_checkpoint(checkpoint_dir=None):
    """Determines if there are checkpoints available."""
    if checkpoint_dir is None:
        checkpoint_dir = get_checkpoint_dir()
    if not os.path.exists(checkpoint_dir):
        return False
    return any(_NAME_PREFIX in f for f in os.listdir(checkpoint_dir))


def save_checkpoint(model, epoch, checkpoint_dir=None, best=False, **kwargs):
    """Saves a checkpoint."""
    if checkpoint_dir is None:
        checkpoint_dir = get_checkpoint_dir()
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    ms = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    sd = ms.state_dict()
    checkpoint = {
        "epoch": epoch,
        "model_state": sd,
    }
    for k,v in kwargs.items():
        vsd = v.state_dict()
        checkpoint[k] = vsd
    # Write the checkpoint
    checkpoint_file = get_checkpoint_name(epoch + 1, checkpoint_dir, best=best)
    torch.save(checkpoint, checkpoint_file)
    return checkpoint_file


def load_checkpoint(checkpoint_file, model):
    """Loads the checkpoint from the given file."""
    err_str = "Checkpoint '{}' not found"
    assert os.path.exists(checkpoint_file), err_str.format(checkpoint_file)
    # Load the checkpoint on CPU to avoid GPU mem spike
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    # Account for the DDP wrapper in the multi-gpu setting
    ms = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    ms.load_state_dict(checkpoint["model_state"])
    # Load the optimizer state (commonly not done when fine-tuning)
    others = {}
    for k,v in checkpoint.items():
        if k not in ["epoch", "model_state"]:
            others[k] = v
    return checkpoint["epoch"], others
