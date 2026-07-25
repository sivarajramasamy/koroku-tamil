import os
import sys
import importlib.util
import torch
import torchaudio

# Dynamically load the original losses.py directly from the StyleTTS2 submodule path to avoid sys.modules conflicts
repo_root = os.environ.get("BOL_REPO", "/content/koroku-tamil")
orig_path = os.path.join(repo_root, "StyleTTS2", "losses.py")

if not os.path.exists(orig_path):
    # Fallback to local sibling path if run outside colab
    orig_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "StyleTTS2", "losses.py")

spec = importlib.util.spec_from_file_location("losses_original_module", orig_path)
losses_original = importlib.util.module_from_spec(spec)
sys.modules["losses_original_module"] = losses_original
spec.loader.exec_module(losses_original)

# Copy all names from original losses to keep complete compatibility
globals().update({k: v for k, v in losses_original.__dict__.items() if not k.startswith('__')})

# Helper function to ensure tensor is 2D [B, T]
def ensure_2d(t):
    if t.ndim == 1:
        return t.unsqueeze(0)
    elif t.ndim == 3 and t.shape[1] == 1:
        return t.squeeze(1)
    elif t.ndim == 3 and t.shape[0] == 1:
        return t.squeeze(0)
    return t

# Override WavLMLoss to handle batch_size 1 squeezes
class WavLMLoss(losses_original.WavLMLoss):
    def forward(self, wav, y_rec):
        wav = ensure_2d(wav)
        y_rec = ensure_2d(y_rec)
        
        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_16 = ensure_2d(wav_16)
            wav_embeddings = self.wavlm(input_values=wav_16, output_hidden_states=True).hidden_states
            
        y_rec_16 = self.resample(y_rec)
        y_rec_16 = ensure_2d(y_rec_16)
        y_rec_embeddings = self.wavlm(input_values=y_rec_16, output_hidden_states=True).hidden_states

        floss = 0
        for er, eg in zip(wav_embeddings, y_rec_embeddings):
            floss += torch.mean(torch.abs(er - eg))
        
        return floss.mean()

    def generator(self, y_rec):
        y_rec = ensure_2d(y_rec)
        y_rec_16 = self.resample(y_rec)
        y_rec_16 = ensure_2d(y_rec_16)
        y_rec_embeddings = self.wavlm(input_values=y_rec_16, output_hidden_states=True).hidden_states
        y_rec_embeddings = torch.stack(y_rec_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)
        y_df_hat_g = self.wd(y_rec_embeddings)
        loss_gen = torch.mean((1-y_df_hat_g)**2)
        
        return loss_gen

    def discriminator(self, wav, y_rec):
        wav = ensure_2d(wav)
        y_rec = ensure_2d(y_rec)
        
        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_16 = ensure_2d(wav_16)
            wav_embeddings = self.wavlm(input_values=wav_16, output_hidden_states=True).hidden_states
            
            y_rec_16 = self.resample(y_rec)
            y_rec_16 = ensure_2d(y_rec_16)
            y_rec_embeddings = self.wavlm(input_values=y_rec_16, output_hidden_states=True).hidden_states

            y_embeddings = torch.stack(wav_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)
            y_rec_embeddings = torch.stack(y_rec_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)

        y_d_rs = self.wd(y_embeddings)
        y_d_gs = self.wd(y_rec_embeddings)
        
        y_df_hat_r, y_df_hat_g = y_d_rs, y_d_gs
        
        r_loss = torch.mean((1-y_df_hat_r)**2)
        g_loss = torch.mean((y_df_hat_g)**2)
        
        loss_disc_f = r_loss + g_loss
                        
        return loss_disc_f.mean()

    def discriminator_forward(self, wav):
        wav = ensure_2d(wav)
        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_16 = ensure_2d(wav_16)
            wav_embeddings = self.wavlm(input_values=wav_16, output_hidden_states=True).hidden_states
            y_embeddings = torch.stack(wav_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)

        y_d_rs = self.wd(y_embeddings)
        
        return y_d_rs

# Monkey patch original class in the loaded module namespace
losses_original.WavLMLoss = WavLMLoss
