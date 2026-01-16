import torch
import torch.nn as nn
import torch.nn.functional as F

class DenoisingAutoencoder(nn.Module):
    def __init__(
        self,
        seq_len: int = 451,
        latent_dim: int = 16,
        hidden_dims=(64, 128, 256),
    ):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dims = list(hidden_dims)
        self.total_scale = 2 ** len(self.hidden_dims)
        self.pad_len = (self.total_scale - seq_len % self.total_scale) % self.total_scale
        
        # ---------- Encoder ----------
        enc_layers = []
        in_c = 1
        for h in self.hidden_dims:
            enc_layers += [
                nn.Conv1d(in_c, h, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(1, h),
                nn.GELU()
            ]
            in_c = h
        self.encoder_conv = nn.Sequential(*enc_layers)
        
        with torch.no_grad():
            dummy = torch.zeros(1, 1, seq_len + self.pad_len)
            flat_dim = self.encoder_conv(dummy).numel()
            self.unflatten_shape = self.encoder_conv(dummy).shape[1:]
        self.encoder_fc = nn.Linear(flat_dim, latent_dim)
        
        # ---------- Decoder ----------
        self.decoder_fc = nn.Linear(latent_dim, flat_dim)
        
        def up_block(in_c, out_c):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
                nn.Conv1d(in_c, out_c, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(1, out_c),
                nn.GELU()
            )
        
        dec_layers = []
        rev_dims = self.hidden_dims[::-1]
        in_c = rev_dims[0]
        for out_c in rev_dims[1:]:
            dec_layers.append(up_block(in_c, out_c))
            in_c = out_c
        dec_layers.append(
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
                nn.Conv1d(in_c, 1, kernel_size=3, stride=1, padding=1)
            )
        )
        self.decoder_conv = nn.Sequential(*dec_layers)

    # ----- helpers -----
    def encode(self, x):
        x = self.encoder_conv(x)
        x = x.flatten(1)
        return self.encoder_fc(x)

    def decode(self, z):
        x = self.decoder_fc(z).view(-1, *self.unflatten_shape)
        x = self.decoder_conv(x)
        if self.pad_len:
            x = x[:, :, :-self.pad_len]
        return x

    def forward(self, x):
        if self.pad_len:
            x = F.pad(x, (0, self.pad_len))
        z = self.encode(x)
        return self.decode(z),z