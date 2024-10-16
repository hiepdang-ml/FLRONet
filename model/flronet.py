from typing import List, Tuple

import torch
import torch.nn as nn

from model.modules import AFNOLayer


class StackedBranchNet(nn.Module):

    def __init__(
        self, 
        n_channels: int,
        n_afno_layers: int,
        embedding_dim: int, 
        block_size: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.n_channels: int = n_channels
        self.n_afno_layers: int = n_afno_layers
        self.embedding_dim: int = embedding_dim
        self.block_size: int = block_size
        self.dropout_rate: float = dropout_rate

        self.embedding_layer = nn.Linear(in_features=n_channels, out_features=embedding_dim)
        self.afno_block = nn.Sequential(
            *[AFNOLayer(embedding_dim=embedding_dim, block_size=block_size, dropout_rate=dropout_rate) for _ in range(n_afno_layers)]
        )
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim * 4),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=embedding_dim * 4, out_features=n_channels),
        )

    def forward(self, sensor_value: torch.Tensor) -> torch.Tensor:
        batch_size, n_timeframes, n_channels, H, W = sensor_value.shape
        assert n_channels == self.n_channels

        output: torch.Tensor = sensor_value.permute(0, 1, 3, 4, 2)
        output = self.embedding_layer(output)   # (batch_size, n_timeframes, H, W, self.embedding_dim)
        output = self.afno_block(output)        # (batch_size, n_timeframes, H, W, self.embedding_dim)
        output = self.mlp(output)               # (batch_size, n_timeframes, H, W, self.n_channels)
        output: torch.Tensor = output.permute(0, 1, 4, 2, 3)
        assert output.shape == sensor_value.shape
        return output


class StackedTrunkNet(nn.Module):

    def __init__(
        self, 
        n_fullstate_timeframes: int,
        n_sensor_timeframes: int, 
        total_timeframes: int,
        resolution: Tuple[int, int],
        dropout_rate: float,
    ):
        super().__init__()
        self.n_fullstate_timeframes: int = n_fullstate_timeframes
        self.n_sensor_timeframes: int = n_sensor_timeframes
        self.total_timeframes: int = total_timeframes
        self.resolution: Tuple[int, int] = resolution
        self.dropout_rate: float = dropout_rate

        self.H, self.W = self.resolution
        self.mlp = nn.Linear(in_features=1, out_features=self.H * self.W)

    def forward(self, fullstate_timeframes: torch.Tensor, sensor_timeframes: torch.Tensor) -> torch.Tensor:
        assert fullstate_timeframes.ndim == sensor_timeframes.ndim == 2
        batch_size: int = sensor_timeframes.shape[0]
        assert fullstate_timeframes.shape == (batch_size, self.n_fullstate_timeframes)
        assert sensor_timeframes.shape == (batch_size, self.n_sensor_timeframes)

        # outer product and normalized to unit scale (range 0-1)
        output: torch.Tensor = fullstate_timeframes.unsqueeze(1) * sensor_timeframes.unsqueeze(2) / (self.total_timeframes ** 2)
        assert output.shape == (batch_size, self.n_sensor_timeframes, self.n_fullstate_timeframes)
        
        # linear transformation
        output = output.unsqueeze(-1).unsqueeze(-1)
        output = self.mlp(output)
        output = output.reshape(batch_size, self.n_sensor_timeframes, self.n_fullstate_timeframes, self.H, self.W)
        return output


class FLRONet(nn.Module):

    def __init__(
        self,
        n_channels: int,
        n_afno_layers: int,
        embedding_dim: int, 
        block_size: int,
        dropout_rate: float,
        n_fullstate_timeframes: int,
        n_sensor_timeframes: int, 
        total_timeframes: int,
        resolution: Tuple[int, int],
        n_stacked_networks: int,
    ):
        super().__init__()
        self.n_channels: int = n_channels
        self.n_afno_layers: int = n_afno_layers
        self.embedding_dim: int = embedding_dim
        self.block_size: int = block_size
        self.dropout_rate: float = dropout_rate
        self.n_fullstate_timeframes: int = n_fullstate_timeframes
        self.n_sensor_timeframes: int = n_sensor_timeframes
        self.total_timeframes: int = total_timeframes
        self.resolution: Tuple[int, int] = resolution
        self.n_stacked_networks: int = n_stacked_networks

        self.H, self.W = self.resolution
        self.branch_nets = nn.ModuleList(
            modules=[
                StackedBranchNet(
                    n_channels=n_channels, n_afno_layers=n_afno_layers, 
                    embedding_dim=embedding_dim, block_size=block_size, 
                    dropout_rate=dropout_rate
                ) 
                for _ in range(n_stacked_networks)
            ]
        )
        self.trunk_nets = nn.ModuleList(
            modules=[
                StackedTrunkNet(
                    n_fullstate_timeframes=n_fullstate_timeframes, 
                    n_sensor_timeframes=n_sensor_timeframes,
                    total_timeframes=total_timeframes, 
                    resolution=resolution, dropout_rate=dropout_rate,
                )
                for _ in range(n_stacked_networks)
            ]
        )
        self.bias = nn.Parameter(data=torch.randn(1))

    def forward(
        self, 
        sensor_timeframes: torch.Tensor, 
        sensor_values: torch.Tensor, 
        fullstate_timeframes: torch.Tensor, 
    ) -> torch.Tensor:
        
        batch_size: int = sensor_timeframes.shape[0]
        assert sensor_timeframes.shape == (batch_size, self.n_sensor_timeframes)
        assert sensor_values.shape == (batch_size, self.n_sensor_timeframes, self.n_channels, self.H, self.W)
        assert fullstate_timeframes.shape == (batch_size, self.n_fullstate_timeframes)
        
        output: torch.Tensor = torch.zeros(
            batch_size, self.n_fullstate_timeframes, self.n_channels, self.H, self.W,
            device=sensor_values.device
        )

        for i in range(self.n_stacked_networks):
            # branch
            branch_net: StackedBranchNet = self.branch_nets[i]
            branch_output: torch.Tensor = branch_net(sensor_values)
            # trunk
            trunk_net: StackedTrunkNet = self.trunk_nets[i]
            trunk_output: torch.Tensor = trunk_net(
                fullstate_timeframes=fullstate_timeframes, 
                sensor_timeframes=sensor_timeframes,
            )
            assert branch_output.shape == (batch_size, self.n_sensor_timeframes, self.n_channels, self.H, self.W)
            assert trunk_output.shape == (batch_size, self.n_sensor_timeframes, self.n_fullstate_timeframes, self.H, self.W)
            output += torch.einsum('nschw,nsfhw->nfchw', branch_output, trunk_output)

        return output + self.bias



if __name__ == '__main__':

    from torch.utils.data import DataLoader
    from cfd.embedding import Mask, Voronoi
    from cfd.sensors import AroundCylinder, LHS
    from cfd.dataset import CFDDataset

    # sensor_generator = LHS(n_sensors=32)
    sensor_generator = AroundCylinder(n_sensors=32)
    # embedding_generator = Mask()
    embedding_generator = Voronoi()

    dataset = CFDDataset(
        root='./data/val', 
        init_sensor_timeframes=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        n_fullstate_timeframes_per_chunk=10,
        resolution=(64, 128),
        sensor_generator=sensor_generator, 
        embedding_generator=embedding_generator,
        seed=1,
    )
    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
    sensor_timeframe_tensor, sensor_tensor, fullstate_timeframe_tensor, fullstate_tensor = next(iter(dataloader))

    device = torch.device('cuda')
    sensor_timeframe_tensor = sensor_timeframe_tensor.to(device)
    sensor_tensor = sensor_tensor.to(device)
    fullstate_timeframe_tensor = fullstate_timeframe_tensor.to(device)
    fullstate_tensor = fullstate_tensor.to(device)

    self = FLRONet(
        n_channels=2, n_afno_layers=1, embedding_dim=256, block_size=16, dropout_rate=0.1, 
        n_fullstate_timeframes=10, n_sensor_timeframes=11, resolution=(64, 128),
        n_stacked_networks=2,
    ).to(device)

    output = self(
        sensor_timeframe_tensor=sensor_timeframe_tensor, 
        sensor_tensor=sensor_tensor, 
        fullstate_timeframe_tensor=fullstate_timeframe_tensor, 
    )
    print(output.shape)
    print(fullstate_tensor.shape)





