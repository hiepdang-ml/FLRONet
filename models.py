from typing import List, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import AFNOLayer


class StackedBranchNet(nn.Module):

    def __init__(
        self, 
        n_channels: int,
        n_afno_layers: int,
        embedding_dim: int, 
        block_size: int,
        dropout_rate: float,
    ):
        self.n_channels: int = n_channels
        self.n_afno_layers: int = n_afno_layers
        self.embedding_dim: int = embedding_dim
        self.block_size: int = block_size
        self.dropout_rate: float = dropout_rate

        self.embedding_layer = nn.Linear(in_features=n_channels, out_features=embedding_dim)
        self.afno_block = nn.Sequential(
            *[AFNOLayer(embedding_dim, block_size, dropout_rate) for _ in range(n_afno_layers)]
        )
        self.mlp = nn.Linear(in_features=embedding_dim, out_features=n_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size, timeframes, n_channels, H, W = input.shape
        assert n_channels == self.n_channels

        output: torch.Tensor = input.permute(0, 1, 3, 4, 2)
        output = self.embedding_layer(output)   # (batch_size, timeframes, H, W, self.embedding_dim)
        output = self.afno_block(output)        # (batch_size, timeframes, H, W, self.embedding_dim)
        output = self.mlp(output)               # (batch_size, timeframes, H, W, self.n_channels)
        output: torch.Tensor = input.permute(0, 1, 4, 2, 3)
        assert output.shape == input.shape
        return output


# TODO: TrunkNet not finished
class TrunkNet(nn.Module):

    def __init__(
        self, 
        n_sensor_timeframes: int, 
        n_fullstate_timeframes: int,
    ):
        self.n_sensor_timeframes: int = n_sensor_timeframes
        self.n_fullstate_timeframes: int = n_fullstate_timeframes

    def forward(input: torch.Tensor) -> torch.Tensor:
        pass




