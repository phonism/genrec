#!/usr/bin/env python3
"""
Convert old checkpoints saved with 'generative_recommenders' module name
to work with the new 'genrec' module name.

Usage:
    python scripts/convert_checkpoint.py <checkpoint_path>

Example:
    python scripts/convert_checkpoint.py out/rqvae/amazon/beauty/checkpoint_epoch_999.pt
"""
import sys
import types
import argparse

# Create module alias BEFORE importing anything else
# This makes 'generative_recommenders' an alias for 'genrec'
class ModuleAlias(types.ModuleType):
    def __init__(self, target_name):
        self._target_name = target_name
        super().__init__(target_name)

    def __getattr__(self, name):
        import importlib
        target = importlib.import_module(self._target_name)
        return getattr(target, name)

# Pre-register the alias
sys.modules['generative_recommenders'] = ModuleAlias('genrec')

# Also need to handle submodules
class SubmoduleFinder:
    def find_module(self, fullname, path=None):
        if fullname.startswith('generative_recommenders.'):
            return self
        return None

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]

        # Map generative_recommenders.X -> genrec.X
        new_name = fullname.replace('generative_recommenders.', 'genrec.', 1)
        import importlib
        module = importlib.import_module(new_name)
        sys.modules[fullname] = module
        return module

sys.meta_path.insert(0, SubmoduleFinder())

# Now safe to import torch and genrec
import torch


def convert_checkpoint(input_path: str, output_path: str = None):
    """
    Convert checkpoint by loading with module alias and re-saving.
    """
    if output_path is None:
        output_path = input_path  # Overwrite original

    print(f"Loading checkpoint from {input_path}...")
    state = torch.load(input_path, map_location='cpu', weights_only=False)

    print(f"Saving converted checkpoint to {output_path}...")
    torch.save(state, output_path)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert old checkpoints to new module name")
    parser.add_argument("checkpoint", help="Path to checkpoint file")
    parser.add_argument("-o", "--output", help="Output path (default: overwrite original)")
    args = parser.parse_args()

    convert_checkpoint(args.checkpoint, args.output)
