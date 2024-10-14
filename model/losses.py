from functools import cached_property
from typing import Tuple, Literal

import torch
import torch.nn as nn


class CustomMSE(nn.Module):

    def __init__(
        self, 
        resolution: Tuple[int, int],
        sensor_positions: torch.Tensor,
        a: float,
        r: float, 
        reduction: Literal['mean', 'sum', 'none']
    ):
        super().__init__()
        self.resolution: Tuple[int, int] = resolution
        self.sensor_positions: torch.Tensor = sensor_positions
        self.a: float = a
        self.r: float = r
        self.reduction: str = reduction
        self.H, self.W = resolution

    def forward(
        self, 
        fullstate_frames: torch.Tensor, 
        reconstructed_frames: torch.Tensor, 
    ) -> torch.Tensor:

        assert fullstate_frames.shape == reconstructed_frames.shape # (batch_size, n_timeframes, n_channels, H, W)
        plain_mse_loss = (fullstate_frames - reconstructed_frames) ** 2
        weighted_mse_loss = plain_mse_loss * self.weights    # broadcasted
        assert weighted_mse_loss.shape == reconstructed_frames.shape

        if self.reduction == 'mean':
            return weighted_mse_loss.mean()
        elif self.reduction == 'sum':
            return weighted_mse_loss.sum()
        else:
            return weighted_mse_loss

    # read-only
    @cached_property
    def weights(self) -> torch.Tensor:
        assert self.sensor_positions.ndim == 2 and self.sensor_positions.shape[1] == 2
        n_sensors: int = self.sensor_positions.shape[0]
        h_coords = torch.arange(self.H, device=self.sensor_positions.device)
        h_coords = h_coords.unsqueeze(1).repeat(1, self.W).unsqueeze(0).repeat(n_sensors, 1, 1) 
        w_coords = torch.arange(self.W, device=self.sensor_positions.device)
        w_coords = w_coords.unsqueeze(0).repeat(self.H, 1).unsqueeze(0).repeat(n_sensors, 1, 1) 
        assert h_coords.shape == w_coords.shape == (n_sensors, self.H, self.W)

        sensor_h = self.sensor_positions[:, 0].unsqueeze(1).unsqueeze(2)
        sensor_w = self.sensor_positions[:, 1].unsqueeze(1).unsqueeze(2)
        assert sensor_h.shape == sensor_w.shape == (n_sensors, 1, 1)

        squared_distances: torch.Tensor = (h_coords - sensor_h) ** 2 + (w_coords - sensor_w) ** 2
        assert squared_distances.shape == (n_sensors, self.H, self.W)
        # weight = a * exp(-s * d^2 ) + 1
        # in which: (a + 1) is the max weight at the sensor position (d=0), s controls the spread (decay rate), min weight is always 1
        weights = self.a * torch.exp(-squared_distances * self.r) + 1
        weights = weights.mean(dim=0) # pixel-level loss does not depend on n_sensors
        assert weights.shape == (self.H, self.W)
        return weights


if __name__ == '__main__':

    from torch.utils.data import DataLoader

    from cfd.dataset import CFDDataset
    from cfd.sensors import LHS, AroundCylinder
    from cfd.embedding import Mask, Voronoi
    
    # sensor_generator = LHS(spatial_shape=(140, 240), n_sensors=32)
    sensor_generator = AroundCylinder(resolution=(140, 240), n_sensors=64)
    # embedding_generator = Mask()
    embedding_generator = Voronoi(weighted=False)

    dataset = CFDDataset(
        root='./data/val', 
        init_sensor_timeframe_indices=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        n_fullstate_timeframes_per_chunk=10,
        n_samplings_per_chunk=1,
        resolution=(140, 240),
        sensor_generator=sensor_generator, 
        embedding_generator=embedding_generator,
        seed=1,
    )

    dataloader = DataLoader(dataset, batch_size=32)
    sensor_timeframe_tensor, sensor_tensor, fullstate_timeframe_tensor, fullstate_tensor = next(iter(dataloader))
    sensor_positions = dataset.sensor_positions

    self = CustomMSE(resolution=(140, 240), sensor_positions=sensor_positions, a=10., r=0.01, reduction='none')
    reconstructed_fullstate_tensor = torch.rand_like(fullstate_tensor)
    loss = self(reconstructed_fullstate_tensor, fullstate_tensor)




