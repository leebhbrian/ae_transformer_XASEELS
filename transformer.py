import torch
import torch.nn as nn

class tfuncondRegressor(nn.Module):
    def __init__(self, seq_len=451, d_lat=16, d_model=128, n_head=4, num_layers=4):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.extra_t = 1

        self.value_proj0  = nn.Linear(1, d_model)       
        self.value_proj1  = nn.Linear(1, d_model)
        self.latent_proj = nn.Linear(d_lat, d_model)

        self.pos_emb = nn.Parameter(
            torch.zeros(1, seq_len + seq_len + self.extra_t, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head,
            dim_feedforward=4 * d_model, dropout=0.1,
            activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer,num_layers=num_layers)

        self.head_bv    = nn.Sequential(nn.Linear(d_model, d_model // 2),
                                        nn.GELU(),
                                        nn.Linear(d_model // 2, 1))
        self.head_bader = nn.Sequential(nn.Linear(d_model, d_model // 2),
                                        nn.GELU(),
                                        nn.Linear(d_model // 2, 1))

    def forward(self, inputter,recon, latent):
        inputter_t = self.value_proj0(inputter.unsqueeze(-1)) 
        recon_t = self.value_proj1(recon.unsqueeze(-1))
        lat_t   = self.latent_proj(latent).unsqueeze(1)
        tokens  = torch.cat([lat_t, inputter_t, recon_t], dim=1)
        tokens  = tokens + self.pos_emb[:, :tokens.size(1), :]

        h       = self.encoder(tokens)
        pooled  = h[:, 0, :]
        out1    = self.head_bv(pooled).squeeze(-1)
        out2    = self.head_bader(pooled).squeeze(-1)
        return torch.stack([out1, out2], dim=1)
