from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):

    def __init__(self, embedding_dim: int, n_hmodes: int, n_wmodes: int):
        super().__init__()
        self.embedding_dim: int = embedding_dim
        self.n_hmodes: int = n_hmodes
        self.n_wmodes: int = n_wmodes
        self.scale: float = 0.02
        self.weights_real = nn.Parameter(
            self.scale * torch.randn(2, embedding_dim, embedding_dim, n_hmodes, n_wmodes, dtype=torch.float)
        )
        self.weights_imag = nn.Parameter(
            self.scale * torch.randn(2, embedding_dim, embedding_dim, n_hmodes, n_wmodes, dtype=torch.float)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 4
        n_frames, embedding_dim, H, W = input.shape
        assert embedding_dim == self.embedding_dim

        padded_H: int = self.next_power_of_2(H)
        padded_W: int = self.next_power_of_2(W)
        padded_input: torch.Tensor = F.pad(input=input, pad=(0, padded_W - W, 0, padded_H - H), mode='constant', value=0)
        # FFT
        fourier_coeff: torch.Tensor = torch.fft.rfft2(padded_input, dim=(2, 3), norm="ortho")
        output_real = torch.zeros((n_frames, embedding_dim, H, W), device='cuda')
        output_imag = torch.zeros((n_frames, embedding_dim, H, W), device='cuda')

        pos_freq_slice: Tuple[slice, slice, slice, slice] = (
            slice(None), slice(None), slice(None, self.n_hmodes), slice(None, self.n_wmodes)
        )   # [:, :, :self.n_hmodes, :self.n_wmodes] 
        neg_freq_slice: Tuple[slice, slice, slice, slice] = (
            slice(None), slice(None), slice(-self.n_hmodes, None), slice(None, self.n_wmodes)
        )   # [:, :, -self.n_hmodes:, :self.n_wmodes]
        output_real[pos_freq_slice], output_imag[pos_freq_slice] = self.complex_mul(
            input_real=fourier_coeff.real[pos_freq_slice], 
            input_imag=fourier_coeff.imag[pos_freq_slice],
            weights_real=self.weights_real[0],
            weights_imag=self.weights_imag[0],
        )
        output_real[neg_freq_slice], output_imag[neg_freq_slice] = self.complex_mul(
            input_real=fourier_coeff.real[neg_freq_slice], 
            input_imag=fourier_coeff.imag[neg_freq_slice],
            weights_real=self.weights_real[1],
            weights_imag=self.weights_imag[1],
        )
        # IFFT
        output: torch.Tensor = torch.complex(real=output_real, imag=output_imag)
        output = torch.fft.irfft2(input=output, s=(H, W), dim=(2, 3), norm="ortho")
        assert output.shape == input.shape == (n_frames, embedding_dim, H, W)
        return output

    @staticmethod
    def complex_mul(
        input_real: torch.Tensor,
        input_imag: torch.Tensor,
        weights_real: torch.Tensor,
        weights_imag: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ops: str = 'nihw,iohw->nohw'
        real_part: torch.Tensor = (
            torch.einsum(ops, input_real, weights_real) - torch.einsum(ops, input_imag, weights_imag)
        )
        imag_part: torch.Tensor = (
            torch.einsum(ops, input_real, weights_imag) + torch.einsum(ops, input_imag, weights_real)
        )
        return real_part, imag_part

    @staticmethod
    def next_power_of_2(x: int) -> int:
        return 1 if x == 0 else 2 ** (x - 1).bit_length()


class FNOBranchNet(nn.Module):

    def __init__(self, n_channels: int, n_fno_layers: int, n_hmodes: int, n_wmodes: int, embedding_dim: int):
        super().__init__()
        self.n_channels: int = n_channels
        self.n_fno_layers: int = n_fno_layers
        self.n_hmodes: int = n_hmodes
        self.n_wmodes: int = n_wmodes
        self.embedding_dim: int = embedding_dim

        self.embedding_layer = nn.Sequential(
            nn.Linear(in_features=n_channels, out_features=128),
            nn.GELU(),
            nn.Linear(in_features=128, out_features=256),
            nn.GELU(),
            nn.Linear(in_features=256, out_features=embedding_dim),
        )
        self.spectral_conv_layers = nn.ModuleList(
            modules=[SpectralConv2d(embedding_dim=embedding_dim, n_hmodes=n_hmodes, n_wmodes=n_wmodes) for _ in range(n_fno_layers)]
        )
        self.Ws = nn.ModuleList(
            modules=[
                nn.Conv2d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=1)
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

    def forward(self, sensor_value: torch.Tensor, out_resolution: Tuple[int, int]) -> torch.Tensor:
        batch_size, in_timeframes, n_channels, in_H, in_W = sensor_value.shape
        assert n_channels == self.n_channels
        flattened_sensor_value: torch.Tensor = sensor_value.flatten(start_dim=0, end_dim=1)
        # embedding
        embedding: torch.Tensor = self.embedding_layer(flattened_sensor_value.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # interpolate embeddings to output resolution
        output: torch.Tensor = F.interpolate(input=embedding, size=out_resolution) if out_resolution != (in_H, in_W) else embedding
        # fno
        for i in range(self.n_fno_layers):
            spectral_conv_layer: SpectralConv2d = self.spectral_conv_layers[i]
            out1: torch.Tensor = spectral_conv_layer(input=output)
            W: nn.Conv2d = self.Ws[i]
            out2: torch.Tensor = W(input=output)
            output = out1 + out2
            if i < self.n_fno_layers - 1:   # not the last layer
                output = F.gelu(output)

        # decoding
        output = self.decoder(output.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out_H, out_W = out_resolution
        output = output.reshape(batch_size, in_timeframes, n_channels, out_H, out_W)
        return output
    

class UNetBranchNet(nn.Module):

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
        batch_size, in_timesteps, n_channels, H, W = input.shape
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
        return reshaped_output.reshape(batch_size, in_timesteps, self.n_channels, H, W)
    
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


class MLPBranchNet(nn.Module):

    def __init__(self, n_channels: int, embedding_dim: int, n_sensors: int, resolution: Tuple[int, int]):
        super().__init__()
        self.n_channels: int = n_channels
        self.embedding_dim: int = embedding_dim
        self.n_sensors: int = n_sensors
        self.resolution: Tuple[int, int] = resolution
        self.H, self.W = resolution
        
        hidden_dim: int = embedding_dim * n_sensors
        block0 = nn.Sequential(
            nn.Linear(in_features=n_channels * n_sensors, out_features=hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )
        block1 = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )
        block2 = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )
        block3 = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )
        block4 = nn.Linear(in_features=hidden_dim, out_features=n_channels * self.H * self.W)
        self.blocks = nn.Sequential(block0, block1, block2, block3, block4)

    def forward(self, input: torch.Tensor):
        assert input.ndim == 4
        batch_size, in_timesteps, n_channels, n_sensors = input.shape
        assert n_channels == self.n_channels
        assert n_sensors == self.n_sensors
        
        output: torch.Tensor = input.flatten(start_dim=2, end_dim=3)
        assert output.shape == (batch_size, in_timesteps, n_channels * n_sensors)
        output = self.blocks(output)
        assert output.shape == (batch_size, in_timesteps, n_channels * self.H * self.W)
        return output
    

class SinusoidEmbedding(nn.Module):

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim: int = embedding_dim

        # Frequency scaling
        self.w = 1. / torch.pow(
            input=torch.tensor(10_000., dtype=torch.float, device='cuda'),
            exponent=torch.arange(0, embedding_dim, 2, dtype=torch.float, device='cuda') / embedding_dim,
        )
        assert self.w.shape == (embedding_dim // 2,)

    def forward(self, timeframes: torch.Tensor) -> torch.Tensor:
        assert timeframes.ndim == 2
        batch_size, n_timeframes = timeframes.shape
        timeframes = timeframes.unsqueeze(-1)  # (batch_size, n_timeframes, 1)
        sinusoid = torch.zeros(*timeframes.shape[:-1], self.embedding_dim, device='cuda')
        sinusoid[:, :, 0::2] = torch.sin(timeframes * self.w)
        sinusoid[:, :, 1::2] = torch.cos(timeframes * self.w)
        assert sinusoid.shape == (batch_size, n_timeframes, self.embedding_dim)
        return sinusoid


class TrunkNet(nn.Module):

    def __init__(self, embedding_dim: int, n_outputs: int):
        super().__init__()
        self.embedding_dim: int = embedding_dim
        self.n_outputs: int = n_outputs
        self.mlps = nn.ModuleList(
            modules=[
                nn.Sequential(
                    nn.Linear(in_features=embedding_dim, out_features=embedding_dim),
                    nn.ReLU(),
                    nn.Linear(in_features=embedding_dim, out_features=embedding_dim),
                    nn.ReLU(),
                    nn.Linear(in_features=embedding_dim, out_features=embedding_dim),
                )
                for _ in range(n_outputs)
            ]
        )

    def forward(self, fullstate_time_embeddings: torch.Tensor, sensor_time_embeddings: torch.Tensor) -> List[torch.Tensor]:
        assert fullstate_time_embeddings.ndim == sensor_time_embeddings.ndim == 3
        assert fullstate_time_embeddings.shape[0] == sensor_time_embeddings.shape[0]
        assert fullstate_time_embeddings.shape[2] == sensor_time_embeddings.shape[2] == self.embedding_dim
        batch_size: int = fullstate_time_embeddings.shape[0]
        n_fullstate_timeframes: int = fullstate_time_embeddings.shape[1]
        n_sensor_timeframes: int = sensor_time_embeddings.shape[1]

        outputs: List[torch.Tensor] = []
        for i in range(self.n_outputs):
            mlp: nn.Module = self.mlps[i]
            output: torch.Tensor = torch.einsum('nse,nfe->nsf', mlp(sensor_time_embeddings), mlp(fullstate_time_embeddings))
            assert output.shape == (batch_size, n_sensor_timeframes, n_fullstate_timeframes)
            outputs.append(output)

        return outputs


class _BaseFLRONet(nn.Module):

    def __init__(self, n_channels: int, embedding_dim: int, n_stacked_networks: int):
        super().__init__()
        self.n_channels: int = n_channels
        self.embedding_dim: int = embedding_dim
        self.n_stacked_networks: int = n_stacked_networks
        # Trunk net
        self.sinusoid_embedding = SinusoidEmbedding(embedding_dim=embedding_dim)
        self.trunk_net = TrunkNet(embedding_dim=embedding_dim, n_outputs=n_stacked_networks)
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
        if isinstance(self, FLRONetMLP):
            assert sensor_values.ndim == 4
            n_sensors: int = sensor_values.shape[-1]
            assert sensor_values.shape == (batch_size, n_sensor_timeframes, self.n_channels, n_sensors)
        else:
            assert sensor_values.ndim == 5
            in_H, in_W = sensor_values.shape[-2:]
            assert sensor_values.shape == (batch_size, n_sensor_timeframes, self.n_channels, in_H, in_W)

        out_H, out_W = out_resolution
        # TrunkNet
        fullstate_time_embeddings: torch.Tensor = self.sinusoid_embedding(timeframes=fullstate_timeframes)
        sensor_time_embeddings: torch.Tensor = self.sinusoid_embedding(timeframes=sensor_timeframes)
        trunk_outputs: List[torch.Tensor] = self.trunk_net(
            fullstate_time_embeddings=fullstate_time_embeddings, sensor_time_embeddings=sensor_time_embeddings
        )
        assert len(trunk_outputs) == self.n_stacked_networks

        # BranchNet
        branch_outputs: List[torch.Tensor] = []
        for i in range(self.n_stacked_networks):
            branch_net: FNOBranchNet | UNetBranchNet | MLPBranchNet = self.branch_nets[i]
            if isinstance(branch_net, FNOBranchNet):
                branch_output: torch.Tensor = branch_net(sensor_value=sensor_values, out_resolution=out_resolution)
            else:
                branch_output: torch.Tensor = branch_net(sensor_value=sensor_values)
            assert branch_output.shape == (batch_size, n_sensor_timeframes, self.n_channels, out_H, out_W)
            branch_outputs.append(branch_output)

        # Fusing (Product)
        output: torch.Tensor = torch.zeros(
            batch_size, n_fullstate_timeframes, self.n_channels, out_H, out_W,
            device='cuda'
        )
        for i in range(self.n_stacked_networks):
            output += torch.einsum('nschw,nsf->nfchw', branch_outputs[i], trunk_outputs[i])

        return output + self.bias

    def freeze_branchnets(self):
        for branch_net in self.branch_nets:
            for param in branch_net.parameters():
                param.requires_grad = False

    def freeze_trunknets(self):
        for trunk_net in self.trunk_nets:
            for param in trunk_net.parameters():
                param.requires_grad = False

    def freeze_bias(self):
        self.bias.requires_grad = False


class FLRONetFNO(_BaseFLRONet):

    def __init__(
        self,
        n_channels: int, n_fno_layers: int, n_hmodes: int, n_wmodes: int, 
        embedding_dim: int, n_stacked_networks: int,
    ):
        super().__init__(n_channels=n_channels, embedding_dim=embedding_dim, n_stacked_networks=n_stacked_networks)
        self.n_fno_layers: int = n_fno_layers
        self.n_hmodes: int = n_hmodes
        self.n_wmodes: int = n_wmodes

        self.branch_nets = nn.ModuleList(
            modules=[
                FNOBranchNet(
                    n_channels=n_channels, n_fno_layers=n_fno_layers, n_hmodes=n_hmodes, n_wmodes=n_wmodes, embedding_dim=embedding_dim,
                )
                for _ in range(n_stacked_networks)
            ]
        )


class FLRONetMLP(_BaseFLRONet):

    def __init__(self, n_channels: int, embedding_dim: int, n_sensors: int, resolution: int, n_stacked_networks: int):
        super().__init__(n_channels=n_channels, embedding_dim=embedding_dim, n_stacked_networks=n_stacked_networks)
        self.n_sensors: int = n_sensors
        self.resolution: int = resolution

        self.branch_nets = nn.ModuleList(
            modules=[
                MLPBranchNet(n_channels=n_channels, embedding_dim=embedding_dim, n_sensors=n_sensors, resolution=resolution)
                for _ in range(n_stacked_networks)
            ]
        )


class FLRONetUNet(_BaseFLRONet):

    def __init__(self, n_channels: int, embedding_dim: int, n_stacked_networks: int):
        super().__init__(n_channels=n_channels, embedding_dim=embedding_dim, n_stacked_networks=n_stacked_networks)

        self.branch_nets = nn.ModuleList(
            modules=[
                UNetBranchNet(n_channels=n_channels, embedding_dim=embedding_dim)
                for _ in range(n_stacked_networks)
            ]
        )

