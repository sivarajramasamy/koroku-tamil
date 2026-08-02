import os
import sys
import importlib.util

# Load the original models.py from the StyleTTS2 submodule path to avoid sys.modules conflicts
repo_root = os.environ.get("BOL_REPO", "/content/koroku-tamil")
orig_path = os.path.join(repo_root, "StyleTTS2", "models.py")
if not os.path.exists(orig_path):
    orig_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "StyleTTS2", "models.py")

spec = importlib.util.spec_from_file_location("models_original_module", orig_path)
models_original = importlib.util.module_from_spec(spec)
sys.modules["models_original_module"] = models_original
spec.loader.exec_module(models_original)

# Copy all names to keep complete compatibility
globals().update({k: v for k, v in models_original.__dict__.items() if not k.startswith('__')})

# Save original load_checkpoint
orig_load_checkpoint = models_original.load_checkpoint

# Define patched load_checkpoint
def load_checkpoint(model, optimizer, path, load_only_params=True, ignore_modules=[]):
    model, optimizer, epoch, iters = orig_load_checkpoint(model, optimizer, path, load_only_params, ignore_modules)
    
    # If resuming a Stage 2 checkpoint to continue training, increment the epoch by 1 so we skip the completed epoch
    if not load_only_params and "epoch_2nd" in path:
        print(f"[Patch] Resuming Stage 2 checkpoint {path}. Incrementing start epoch from {epoch} to {epoch + 1} to skip the completed epoch.")
        epoch += 1
        
    return model, optimizer, epoch, iters

# Override in original module
models_original.load_checkpoint = load_checkpoint
