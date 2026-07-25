import os
import sys
import importlib.util

# Dynamically load the original losses.py directly from the StyleTTS2 submodule path to avoid sys.modules conflicts
repo_root = os.environ.get("BOL_REPO", "/content/koroku-tamil")
orig_path = os.path.join(repo_root, "StyleTTS2", "losses.py")

if not os.path.exists(orig_path):
    orig_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "StyleTTS2", "losses.py")

spec = importlib.util.spec_from_file_location("losses_original_module", orig_path)
losses_original = importlib.util.module_from_spec(spec)
sys.modules["losses_original_module"] = losses_original
spec.loader.exec_module(losses_original)

# Copy all names from original losses to keep complete compatibility
globals().update({k: v for k, v in losses_original.__dict__.items() if not k.startswith('__')})

# Save original methods
orig_forward = losses_original.WavLMLoss.forward
orig_generator = losses_original.WavLMLoss.generator
orig_discriminator = losses_original.WavLMLoss.discriminator
orig_discriminator_forward = losses_original.WavLMLoss.discriminator_forward

# Helper function to ensure tensor is 2D [B, T]
def ensure_2d(t):
    if t.ndim == 1:
        return t.unsqueeze(0)
    elif t.ndim == 3 and t.shape[1] == 1:
        return t.squeeze(1)
    elif t.ndim == 3 and t.shape[0] == 1:
        return t.squeeze(0)
    return t

# Define wrapper methods that sanitize shapes
def new_forward(self, wav, y_rec):
    return orig_forward(self, ensure_2d(wav), ensure_2d(y_rec))

def new_generator(self, y_rec):
    return orig_generator(self, ensure_2d(y_rec))

def new_discriminator(self, wav, y_rec):
    return orig_discriminator(self, ensure_2d(wav), ensure_2d(y_rec))

def new_discriminator_forward(self, wav):
    return orig_discriminator_forward(self, ensure_2d(wav))

# Replace original class methods
losses_original.WavLMLoss.forward = new_forward
losses_original.WavLMLoss.generator = new_generator
losses_original.WavLMLoss.discriminator = new_discriminator
losses_original.WavLMLoss.discriminator_forward = new_discriminator_forward

# Ensure local module namespace points to the patched class
WavLMLoss = losses_original.WavLMLoss
