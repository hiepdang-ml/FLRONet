from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# DONE
class DropPath(nn.Module):
    
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.p == 0 or not self.training:
            return input    # same as nn.Identity() in .eval() mode
        
        shape: Tuple[int, ...] = (input.shape[0],) + (1,) * (input.ndim - 1)
        random_tensor: torch.Tensor = (1 - self.p) + torch.rand(shape, dtype=input.dtype, device=input.device)
        mask: torch.Tensor = random_tensor.floor()
        output = input.div(1 - self.p) * mask
        return output

# DONE
class AFNOLayer(nn.Module):

    def __init__(
        self, 
        embedding_dim: int, 
        block_size: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.embedding_dim: int = embedding_dim
        self.block_size: int = block_size
        self.dropout_rate: float = dropout_rate
        self.n_blocks: int = self.embedding_dim // self.block_size

        self.scale: float = 0.02
        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.n_blocks, block_size, block_size))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, 1, 1, 1, self.n_blocks, block_size))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.n_blocks, block_size, block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, 1, 1, 1, self.n_blocks, block_size))

        self.ln1 = nn.LayerNorm(normalized_shape=embedding_dim)
        self.ln2 = nn.LayerNorm(normalized_shape=embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=embedding_dim * 4, out_features=embedding_dim),
            nn.Dropout(p=dropout_rate),
        )
        self.droppath = DropPath(p=dropout_rate)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 5
        batch_size, n_timesteps, H, W, embedding_dim = input.shape
        assert embedding_dim == self.embedding_dim

        output: torch.Tensor = self.ln1(input)
        # Fourier transform (Token mixing)
        fourier_coeff: torch.Tensor = torch.fft.rfft2(input=output, dim=(2, 3), norm="ortho")
        # Linear transformation with shared weight (Channel mixing)
        x_modes = fourier_coeff.shape[2]
        y_modes = fourier_coeff.shape[3]
        assert (x_modes, y_modes) == (H, W // 2 + 1)

        fourier_coeff: torch.Tensor = fourier_coeff.reshape(
            batch_size, n_timesteps, x_modes, y_modes, self.n_blocks, self.block_size
        )

        output1_real = torch.zeros(fourier_coeff.shape, device=input.device)
        output1_imag = torch.zeros(fourier_coeff.shape, device=input.device)
        output2_real = torch.zeros(fourier_coeff.shape, device=input.device)
        output2_imag = torch.zeros(fourier_coeff.shape, device=input.device)

        ops: str = 'btxyni,nio->btxyno'
        output1_real = F.gelu(
            torch.einsum(ops, fourier_coeff.real, self.w1[0]) 
            - torch.einsum(ops, fourier_coeff.imag, self.w1[1]) 
            + self.b1[0]
        )
        output1_imag = F.gelu(
            torch.einsum(ops, fourier_coeff.imag, self.w1[0]) 
            + torch.einsum(ops, fourier_coeff.real, self.w1[1]) 
            + self.b1[1]
        )
        output2_real = (
            torch.einsum(ops, output1_real, self.w2[0]) 
            - torch.einsum(ops, output1_imag, self.w2[1]) 
            + self.b2[0]
        )
        output2_imag = (
            torch.einsum(ops, output1_imag, self.w2[0]) 
            + torch.einsum(ops, output1_real, self.w2[1]) 
            + self.b2[1]
        )
        output: torch.Tensor = torch.stack([output2_real, output2_imag], dim=-1)
        output = F.softshrink(output, lambd=0.01)
        output = torch.view_as_complex(output)
        output = output.reshape(batch_size, n_timesteps, x_modes, y_modes, embedding_dim)

        # Inverse Fourier transform (Token demixing)
        output: torch.Tensor = torch.fft.irfft2(
            input=output, 
            s=(H, W),
            dim=(2, 3), 
            norm="ortho",
        )
        assert output.shape == (
            batch_size, n_timesteps, H, W, embedding_dim
        )
        # Skip connection
        output = output + input
        residual = output
        # MLP
        output = self.ln2(output)
        output = self.mlp(output)
        assert output.shape == input.shape
        # Skip connection + Drop path
        output = self.droppath(output) + residual
        return output   # output.shape == input.shape




