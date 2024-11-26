from typing import Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv3d(nn.Module):

    def __init__(self, embedding_dim: int, n_tmodes: int, n_hmodes: int, n_wmodes: int):
        super().__init__()
        self.embedding_dim: int = embedding_dim
        self.n_tmodes: int = n_tmodes
        self.n_hmodes: int = n_hmodes
        self.n_wmodes: int = n_wmodes
        self.scale: float = 0.02
        self.weights_real = nn.Parameter(
            self.scale * torch.randn(4, embedding_dim, embedding_dim, n_hmodes, n_wmodes, n_tmodes, dtype=torch.float)
        )
        self.weights_imag = nn.Parameter(
            self.scale * torch.randn(4, embedding_dim, embedding_dim, n_hmodes, n_wmodes, n_tmodes, dtype=torch.float)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 5
        n_frames, embedding_dim, T, H, W = input.shape
        assert embedding_dim == self.embedding_dim

        padded_T, padded_H, padded_W = map(self.next_power_of_2, (T, H, W))
        padded_input: torch.Tensor = F.pad(
            input=input, 
            pad=(0, padded_W - W, 0, padded_H - H, 0, padded_T - T), 
            mode='constant', value=0
        )
        padded_input = padded_input.permute(0, 1, 3, 4, 2)  # n_frames, embedding_dim, H, W, T
        # FFT
        fourier_coeff: torch.Tensor = torch.fft.rfftn(padded_input, dim=(2, 3, 4), norm="ortho")
        output_real = torch.zeros_like(fourier_coeff.real)
        output_imag = torch.zeros_like(fourier_coeff.imag)

        slice0: Tuple[slice, slice, slice, slice, slice] = (
            slice(None), slice(None),
            slice(None, self.n_hmodes),
            slice(None, self.n_wmodes),
            slice(None, self.n_tmodes),
        )
        slice1: Tuple[slice, slice, slice, slice, slice] = (
            slice(None), slice(None),
            slice(-self.n_hmodes, None),
            slice(None, self.n_wmodes),
            slice(None, self.n_tmodes),
        )
        slice2: Tuple[slice, slice, slice, slice, slice] = (
            slice(None), slice(None),
            slice(None, self.n_hmodes),
            slice(-self.n_wmodes, None),
            slice(None, self.n_tmodes),
        )
        slice3: Tuple[slice, slice, slice, slice, slice] = (
            slice(None), slice(None),
            slice(-self.n_hmodes, None),
            slice(-self.n_wmodes, None),
            slice(None, self.n_tmodes),
        )
        output_real[slice0], output_imag[slice0] = self.complex_mul(
            input_real=fourier_coeff.real[slice0],
            input_imag=fourier_coeff.imag[slice0],
            weights_real=self.weights_real[0],
            weights_imag=self.weights_imag[0],
        )
        output_real[slice1], output_imag[slice1] = self.complex_mul(
            input_real=fourier_coeff.real[slice1],
            input_imag=fourier_coeff.imag[slice1],
            weights_real=self.weights_real[1],
            weights_imag=self.weights_imag[1],
        )
        output_real[slice2], output_imag[slice2] = self.complex_mul(
            input_real=fourier_coeff.real[slice2],
            input_imag=fourier_coeff.imag[slice2],
            weights_real=self.weights_real[2],
            weights_imag=self.weights_imag[2],
        )
        output_real[slice3], output_imag[slice3] = self.complex_mul(
            input_real=fourier_coeff.real[slice3],
            input_imag=fourier_coeff.imag[slice3],
            weights_real=self.weights_real[3],
            weights_imag=self.weights_imag[3],
        )
        # IFFT
        output: torch.Tensor = torch.complex(real=output_real, imag=output_imag)
        output = torch.fft.irfftn(output, s=(H, W, T), dim=(2, 3, 4), norm="ortho")
        output = output.permute(0, 1, 4, 2, 3)
        assert output.shape == input.shape
        return output

    @staticmethod
    def next_power_of_2(x: int) -> int:
        return 1 if x == 0 else 2 ** (x - 1).bit_length()

    @staticmethod
    def complex_mul(
        input_real: torch.Tensor,
        input_imag: torch.Tensor,
        weights_real: torch.Tensor,
        weights_imag: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ops: str = 'ndhw,iodhw->nodhw'
        real_part: torch.Tensor = (
            torch.einsum(ops, input_real, weights_real) - torch.einsum(ops, input_imag, weights_imag)
        )
        imag_part: torch.Tensor = (
            torch.einsum(ops, input_real, weights_imag) + torch.einsum(ops, input_imag, weights_real)
        )
        return real_part, imag_part


class FNO3D(nn.Module):

    def __init__(self, n_channels: int, n_fno_layers: int, n_hmodes: int, n_wmodes: int, n_tmodes: int, embedding_dim: int):
        super().__init__()
        self.n_channels: int = n_channels
        self.n_fno_layers: int = n_fno_layers
        self.n_hmodes: int = n_hmodes
        self.n_wmodes: int = n_wmodes
        self.n_tmodes: int = n_tmodes
        self.embedding_dim: int = embedding_dim

        self.embedding_layer = nn.Sequential(
            nn.Linear(in_features=n_channels, out_features=128),
            nn.GELU(),
            nn.Linear(in_features=128, out_features=256),
            nn.GELU(),
            nn.Linear(in_features=256, out_features=embedding_dim),
        )
        self.spectral_conv_layers = nn.ModuleList(
            modules=[
                SpectralConv3d(embedding_dim=embedding_dim, n_tmodes=n_tmodes, n_hmodes=n_hmodes, n_wmodes=n_wmodes)
                for _ in range(n_fno_layers)
            ]
        )
        self.Ws = nn.ModuleList(
            modules=[
                nn.Conv3d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=1)
                for _ in range(n_fno_layers)
            ]
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=128),
            nn.GELU(),
            nn.Linear(in_features=128, out_features=128),
            nn.GELU(),
            nn.Linear(in_features=128, out_features=n_channels),
        )

    def forward(self, sensor_value: torch.Tensor, out_resolution: Tuple[int, int, int]) -> torch.Tensor:
        batch_size, in_T, n_channels, in_H, in_W = sensor_value.shape
        assert n_channels == self.n_channels
        # embedding
        embedding: torch.Tensor = self.embedding_layer(sensor_value.permute(0, 1, 3, 4, 2)).permute(0, 4, 1, 2, 3)
        assert embedding.shape == (batch_size, embedding, in_T, in_H, in_W)
        # interpolate embeddings to output resolution
        output: torch.Tensor = F.interpolate(input=embedding, size=out_resolution) if out_resolution != (in_T, in_H, in_W) else embedding
        # fno
        for i in range(self.n_fno_layers):
            spectral_conv_layer: SpectralConv3d = self.spectral_conv_layers[i]
            out1: torch.Tensor = spectral_conv_layer(input=output)
            W: nn.Conv3d = self.Ws[i]
            out2: torch.Tensor = W(input=output)
            output = out1 + out2
            if i < self.n_fno_layers - 1:   # not the last layer
                output = F.gelu(output)

        # decoding
        output = self.decoder(output.permute(0, 2, 3, 4, 1)).permute(0, 1, 4, 2, 3)
        out_T, out_H, out_W = out_resolution
        output = output.reshape(batch_size, out_T, n_channels, out_H, out_W)
        return output
    
    