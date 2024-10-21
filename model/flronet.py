from typing import List, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):

    def __init__(self, n_channels: int, embedding_dim: int):
        super().__init__()
        self.n_channels: int = n_channels
        self.embedding_dim: int = embedding_dim
        # Encoder
        self.enc_conv1 = self.conv_block(in_channels=n_channels, out_channels=embedding_dim)
        self.enc_conv2 = self.conv_block(in_channels=embedding_dim, out_channels=embedding_dim * 2)
        self.enc_conv3 = self.conv_block(in_channels=embedding_dim * 2, out_channels=embedding_dim * 4)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Bottleneck
        self.bottleneck_conv = self.conv_block(in_channels=embedding_dim * 4, out_channels=embedding_dim * 8)
        # Decoder
        self.upconv3 = self.upconv(in_channels=embedding_dim * 8, out_channels=embedding_dim * 4)
        self.dec_conv3 = self.conv_block(in_channels=embedding_dim * 8, out_channels=embedding_dim * 4)
        self.upconv2 = self.upconv(in_channels=embedding_dim * 4, out_channels=embedding_dim * 2)
        self.dec_conv2 = self.conv_block(in_channels=embedding_dim * 4, out_channels=embedding_dim * 2)
        self.upconv1 = self.upconv(in_channels=embedding_dim * 2, out_channels=embedding_dim)
        self.dec_conv1 = self.conv_block(in_channels=embedding_dim * 2, out_channels=embedding_dim)
        # Final convolution
        self.final_conv = nn.Conv2d(in_channels=embedding_dim, out_channels=n_channels, kernel_size=1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 5
        batch_size, n_timesteps, n_channels, H, W = input.shape
        assert n_channels == self.n_channels

        # Layer Norm
        reshaped_input: torch.Tensor = input.flatten(start_dim=0, end_dim=1)
        # Encoder
        enc1: torch.Tensor = self.enc_conv1(reshaped_input)
        enc2: torch.Tensor = self.enc_conv2(self.pool(enc1))
        enc3: torch.Tensor = self.enc_conv3(self.pool(enc2))
        # Bottleneck
        bottleneck: torch.Tensor = self.bottleneck_conv(self.pool(enc3))
        # Decoder
        dec3: torch.Tensor = self.upconv3(bottleneck)
        if dec3.shape[-2:] != enc3.shape[-2:]:  # due to input resolution not a power of 2
            dec3 = F.interpolate(dec3, size=enc3.shape[-2:], mode='bilinear', align_corners=False)
        dec3 = torch.cat(tensors=[dec3, enc3], dim=1)
        dec3 = self.dec_conv3(dec3)
        dec2: torch.Tensor = self.upconv2(dec3)
        dec2 = torch.cat(tensors=[dec2, enc2], dim=1)
        dec2 = self.dec_conv2(dec2)
        dec1: torch.Tensor = self.upconv1(dec2)
        dec1 = torch.cat(tensors=[dec1, enc1], dim=1)
        dec1 = self.dec_conv1(dec1)
        # Final output
        reshaped_output: torch.Tensor = self.final_conv(dec1)
        assert reshaped_output.shape == reshaped_input.shape
        output: torch.Tensor = reshaped_output.reshape(batch_size, n_timesteps, self.n_channels, H, W)
        return output
    
    def conv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )

    def upconv(self, in_channels: int, out_channels: int) -> nn.Module:
        return nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)


class FNOLayer(nn.Module):

    def __init__(self, embedding_dim: int, n_modes: int):
        super().__init__()
        self.embedding_dim: int = embedding_dim
        self.n_modes: int = n_modes
        self.scale: float = 0.02
        self.weights_real = nn.Parameter(
            self.scale * torch.randn(2, embedding_dim, embedding_dim, self.n_modes, self.n_modes, dtype=torch.float)
        )
        self.weights_imag = nn.Parameter(
            self.scale * torch.randn(2, embedding_dim, embedding_dim, self.n_modes, self.n_modes, dtype=torch.float)
        )
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels=embedding_dim, out_channels=embedding_dim * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels=embedding_dim * 4, out_channels=embedding_dim, kernel_size=1),
        )
        self.norm = nn.InstanceNorm2d(num_features=embedding_dim)

    def forward(self, input: torch.Tensor, out_resolution: Tuple[int, int]) -> torch.Tensor:
        assert input.ndim == 4
        n_frames, embedding_dim, in_H, in_W = input.shape
        assert embedding_dim == self.embedding_dim
        out_H, out_W = out_resolution
        
        with torch.autocast(device_type="cuda", enabled=False):
            input = input.float()
            fourier_coeff: torch.Tensor = torch.fft.rfft2(input=self.norm(input), dim=(2, 3), norm="ortho")
            assert fourier_coeff.shape == (n_frames, embedding_dim, in_H, in_W // 2 + 1)
    
        output_real: torch.Tensor = torch.zeros(n_frames, embedding_dim, out_H, out_W).cuda()
        output_imag: torch.Tensor = torch.zeros(n_frames, embedding_dim, out_H, out_W).cuda()
        pos_freq_slice: Tuple[slice, slice, slice, slice] = (
            slice(None), slice(None), slice(None, self.n_modes), slice(None, self.n_modes)
        )   # [:, :, :self.n_modes, :self.n_modes] 
        neg_freq_slice: Tuple[slice, slice, slice, slice] = (
            slice(None), slice(None), slice(-self.n_modes, None), slice(None, self.n_modes)
        )   # [:, :, -self.n_modes:, :self.n_modes]
        output_real[pos_freq_slice], output_imag[pos_freq_slice] = self.complex_mul(
            fourier_coeff.real[pos_freq_slice], fourier_coeff.imag[pos_freq_slice],
            self.weights_real[0],
            self.weights_imag[0],
        )
        output_real[neg_freq_slice], output_imag[neg_freq_slice] = self.complex_mul(
            fourier_coeff.real[neg_freq_slice], fourier_coeff.imag[neg_freq_slice],
            self.weights_real[1],
            self.weights_imag[1],
        )
        output: torch.Tensor = torch.complex(real=output_real, imag=output_imag)
        output = torch.fft.irfft2(input=output, s=out_resolution, dim=(2, 3), norm="ortho")
        output = self.mlp(output)
        assert output.shape == (n_frames, embedding_dim, out_H, out_W)
        if (out_H, out_W) == (in_H, in_W):
            output = input + output
        else:
            output = F.interpolate(input, size=(out_H, out_W), mode='bicubic') + output

        return output

    def complex_mul(
        self,
        input_real: torch.Tensor,
        input_imag: torch.Tensor,
        weights_real: torch.Tensor,
        weights_imag: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ops: str = 'nixy,ioxy->noxy'
        real_part: torch.Tensor = (
            torch.einsum(ops, input_real, weights_real) - torch.einsum(ops, input_imag, weights_imag)
        )
        imag_part: torch.Tensor = (
            torch.einsum(ops, input_real, weights_imag) + torch.einsum(ops, input_imag, weights_real)
        )
        return real_part, imag_part


class StackedFNOBranchNet(nn.Module):

    def __init__(self, n_channels: int, n_fno_layers: int, n_fno_modes: int, embedding_dim: int):
        super().__init__()
        self.n_channels: int = n_channels
        self.n_fno_layers: int = n_fno_layers
        self.n_fno_modes: int = n_fno_modes
        self.embedding_dim: int = embedding_dim

        self.updim = nn.Conv2d(in_channels=n_channels, out_channels=embedding_dim, kernel_size=1)
        self.fno_layers = nn.ModuleList(
            modules=[FNOLayer(embedding_dim=embedding_dim, n_modes=n_fno_modes) for _ in range(n_fno_layers)]
        )
        self.downdim = nn.Conv2d(in_channels=embedding_dim, out_channels=n_channels, kernel_size=1)

    def forward(self, sensor_value: torch.Tensor, out_resolution: Tuple[int, int]) -> torch.Tensor:
        batch_size, n_timeframes, n_channels, in_H, in_W = sensor_value.shape
        assert n_channels == self.n_channels
        output: torch.Tensor = sensor_value.flatten(start_dim=0, end_dim=1)
        output = self.updim(output)
        for fno_layer in self.fno_layers:
            output = fno_layer(output, out_resolution=out_resolution)

        output = self.downdim(output)
        out_H, out_W = out_resolution
        output = output.reshape(batch_size, n_timeframes, n_channels, out_H, out_W)
        return output
    

class StackedUNetranchNet(nn.Module):

    def __init__(self, n_channels: int, embedding_dim: int):
        super().__init__()
        self.n_channels: int = n_channels
        self.embedding_dim: int = embedding_dim
        self.embedding_layer = nn.Linear(in_features=n_channels, out_features=embedding_dim)
        self.unet = UNet(n_channels=n_channels, embedding_dim=embedding_dim)
        self.mlp = nn.Linear(in_features=embedding_dim, out_features=n_channels)

    def forward(self, sensor_value: torch.Tensor) -> torch.Tensor:
        batch_size, n_timeframes, n_channels, H, W = sensor_value.shape
        assert n_channels == self.n_channels
        output: torch.Tensor = self.unet(sensor_value)
        assert output.shape == (batch_size, n_timeframes, n_channels, H, W)
        return output


class StackedTrunkNet(nn.Module):

    def __init__(self, embedding_dim: int, total_timeframes: int):
        super().__init__()
        self.total_timeframes: int = total_timeframes
        self.embedding_dim: int = embedding_dim
        self.time_embedding = nn.Embedding(num_embeddings=total_timeframes, embedding_dim=embedding_dim, max_norm=1.)
        nn.init.uniform_(tensor=self.time_embedding.weight, a=0, b=1)

    def forward(self, fullstate_timeframes: torch.LongTensor, sensor_timeframes: torch.LongTensor) -> torch.Tensor:
        assert fullstate_timeframes.ndim == sensor_timeframes.ndim == 2
        batch_size, n_fullstate_timeframes = fullstate_timeframes.shape
        n_sensor_timeframes: int = sensor_timeframes.shape[1]
        # compute temporal embeddings
        fullstate_time_embeddings: torch.Tensor = self.time_embedding(input=fullstate_timeframes)
        assert fullstate_time_embeddings.shape == (batch_size, n_fullstate_timeframes, self.embedding_dim)
        sensor_time_embeddings: torch.Tensor = self.time_embedding(input=sensor_timeframes)
        assert sensor_time_embeddings.shape == (batch_size, n_sensor_timeframes, self.embedding_dim)
        # condition fullstate time on sensor time
        output: torch.Tensor = torch.einsum('nse,nfe->nsf', sensor_time_embeddings, fullstate_time_embeddings)
        assert output.shape == (batch_size, n_sensor_timeframes, n_fullstate_timeframes)
        return output


class FLRONetWithFNO(nn.Module):

    def __init__(
        self,
        n_channels: int, n_fno_layers: int, n_fno_modes: int, 
        embedding_dim: int, total_timeframes: int, n_stacked_networks: int,
    ):
        super().__init__()
        self.n_channels: int = n_channels
        self.n_fno_layers: int = n_fno_layers
        self.n_fno_modes: int = n_fno_modes
        self.embedding_dim: int = embedding_dim
        self.total_timeframes: int = total_timeframes
        self.n_stacked_networks: int = n_stacked_networks

        self.branch_nets = nn.ModuleList(
            modules=[
                StackedFNOBranchNet(
                    n_channels=n_channels, n_fno_layers=n_fno_layers, n_fno_modes=n_fno_modes, embedding_dim=embedding_dim,
                )
                for _ in range(n_stacked_networks)
            ]
        )
        self.trunk_nets = nn.ModuleList(
            modules=[
                StackedTrunkNet(embedding_dim=embedding_dim, total_timeframes=total_timeframes)
                for _ in range(n_stacked_networks)
            ]
        )
        self.bias = nn.Parameter(data=torch.randn(n_channels, 1, 1))

    def forward(
        self, 
        sensor_timeframes: torch.Tensor, 
        sensor_values: torch.Tensor, 
        fullstate_timeframes: torch.Tensor, 
        out_resolution: Tuple[int, int],
    ) -> torch.Tensor:
        assert sensor_timeframes.ndim == fullstate_timeframes.ndim == 2
        assert sensor_timeframes.shape[0] == sensor_values.shape[0] == fullstate_timeframes.shape[0]
        batch_size, n_sensor_timeframes = sensor_timeframes.shape
        n_fullstate_timeframes: int = fullstate_timeframes.shape[1]
        assert sensor_values.ndim == 5

        in_H, in_W = sensor_values.shape[-2:]
        out_H, out_W = out_resolution
        assert sensor_values.shape == (batch_size, n_sensor_timeframes, self.n_channels, in_H, in_W)
        
        output: torch.Tensor = torch.zeros(
            batch_size, n_fullstate_timeframes, self.n_channels, out_H, out_W,
            device=sensor_values.device
        )
        for i in range(self.n_stacked_networks):
            # branch
            branch_net: StackedFNOBranchNet = self.branch_nets[i]
            branch_output: torch.Tensor = branch_net(sensor_value=sensor_values, out_resolution=out_resolution)
            assert branch_output.shape == (batch_size, n_sensor_timeframes, self.n_channels, out_H, out_W)
            # trunk
            trunk_net: StackedTrunkNet = self.trunk_nets[i]
            trunk_output: torch.Tensor = trunk_net(
                fullstate_timeframes=fullstate_timeframes, 
                sensor_timeframes=sensor_timeframes,
            )
            assert trunk_output.shape == (batch_size, n_sensor_timeframes, n_fullstate_timeframes)
            output += torch.einsum('nschw,nsf->nfchw', branch_output, trunk_output)

        return output + self.bias


class FLRONetWithUNet(nn.Module):

    def __init__(self, n_channels: int, embedding_dim: int, total_timeframes: int, n_stacked_networks: int):
        super().__init__()
        self.n_channels: int = n_channels
        self.embedding_dim: int = embedding_dim
        self.total_timeframes: int = total_timeframes
        self.n_stacked_networks: int = n_stacked_networks

        self.branch_nets = nn.ModuleList(
            modules=[
                StackedUNetranchNet(n_channels=n_channels, embedding_dim=embedding_dim)
                for _ in range(n_stacked_networks)
            ]
        )
        self.trunk_nets = nn.ModuleList(
            modules=[
                StackedTrunkNet(embedding_dim=embedding_dim, total_timeframes=total_timeframes)
                for _ in range(n_stacked_networks)
            ]
        )
        self.bias = nn.Parameter(data=torch.randn(n_channels, 1, 1))

    def forward(
        self, 
        sensor_timeframes: torch.Tensor, 
        sensor_values: torch.Tensor, 
        fullstate_timeframes: torch.Tensor, 
    ) -> torch.Tensor:
        assert sensor_timeframes.ndim == fullstate_timeframes.ndim == 2
        assert sensor_timeframes.shape[0] == sensor_values.shape[0] == fullstate_timeframes.shape[0]
        batch_size, n_sensor_timeframes = sensor_timeframes.shape
        n_fullstate_timeframes: int = fullstate_timeframes.shape[1]
        assert sensor_values.ndim == 5

        H, W = sensor_values.shape[-2:]
        assert sensor_values.shape == (batch_size, n_sensor_timeframes, self.n_channels, H, W)
        
        output: torch.Tensor = torch.zeros(
            batch_size, n_fullstate_timeframes, self.n_channels, H, W,
            device=sensor_values.device
        )
        for i in range(self.n_stacked_networks):
            # branch
            branch_net: StackedFNOBranchNet = self.branch_nets[i]
            branch_output: torch.Tensor = branch_net(sensor_value=sensor_values)
            assert branch_output.shape == (batch_size, n_sensor_timeframes, self.n_channels, H, W)
            # trunk
            trunk_net: StackedTrunkNet = self.trunk_nets[i]
            trunk_output: torch.Tensor = trunk_net(
                fullstate_timeframes=fullstate_timeframes, 
                sensor_timeframes=sensor_timeframes,
            )
            assert trunk_output.shape == (batch_size, n_sensor_timeframes, n_fullstate_timeframes)
            output += torch.einsum('nschw,nsf->nfchw', branch_output, trunk_output)

        return output + self.bias
    