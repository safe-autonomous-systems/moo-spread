"""
DiTMOO: (Diffusion Transformer for Multi-Objective Optimization)
"""

import torch
import torch.nn as nn


class DiTBlock(nn.Module):
    """Single DiT transformer block."""

    def __init__(self, hidden_dim: int, num_heads: int, num_obj: int):
        super().__init__()
        # Timestep embedding
        self.timestep_embedding = nn.Linear(1, hidden_dim)
        # Normalisations
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        # Conditioning cross-attention
        self.cond_proj = nn.Sequential(
            nn.Linear(num_obj, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        # Pointwise feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        # Self-attention
        x = x + self.self_attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]

        # Cross-attention
        c_proj = self.cond_proj(c).unsqueeze(1)  # (B, 1, H)
        t_emb = self.timestep_embedding(t.unsqueeze(-1)).unsqueeze(1) # (B, 1, H)
        c_proj = torch.cat([t_emb, c_proj], dim=1) # (B, 2, H)
        x = x + self.cross_attn(self.ln2(x), c_proj, c_proj)[0] # (B, 1, H)

        # Feed-forward
        x = x + self.ffn(self.ln3(x))
        return x


class DiTMOO(nn.Module):
    """
    DiTMOO: Diffusion Transformer for Multi-Objective Optimization.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the raw input vector.
    num_obj : int
        Dimensionality of the conditioning vector.
    hidden_dim : int, optional
        Transformer hidden dimension. Default is 128.
    num_heads : int, optional
        Number of attention heads. Default is 4.
    num_blocks : int, optional
        How many times to apply the DiT block. Default is 1.
    """

    def __init__(
        self,
        input_dim: int,
        num_obj: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_blocks: int = 1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Stack of DiT blocks
        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_dim, num_heads, num_obj)
                for _ in range(num_blocks)
            ]
        )

        # Final normalisation and projection
        self.ln_out = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, input_dim)
        t : (B,) or (B, 1)
            Scalar timestep per sample.
        c : (B, num_obj)
            Conditioning information.
        """
        # Project input and add timestep embedding
        x = self.input_proj(x).unsqueeze(1)  # (B, 1, H)
        
        # Apply DiT blocks
        for block in self.blocks:
            x = block(x, t, c)

        # Project back to original dimension
        x = self.output_proj(self.ln_out(x).squeeze(1))
        return x