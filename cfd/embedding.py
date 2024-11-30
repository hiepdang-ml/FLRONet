from abc import ABC, abstractmethod
from typing import List, Tuple
from functools import cached_property
import random

import torch


class SensorEmbedding(ABC):

    def __init__(
        self,
        resolution: Tuple[int, int],
        sensor_positions: torch.Tensor, 
        dropout_probabilities: List[float] = [], 
        noise_level: float = 0.,
    ):
        self.resolution: Tuple[int, int] = resolution
        self.H, self.W = self.resolution
        assert sensor_positions.ndim == 2 and sensor_positions.shape[1] == 2
        self.sensor_positions: torch.Tensor = sensor_positions.float()
        assert sum(dropout_probabilities) <= 1, "Dropout probabilities must sum to less than 1"
        self.n_max_dropout_sensors: int = len(dropout_probabilities)
        self.dropout_probabilities: List[float] = [1. - sum(dropout_probabilities)] + dropout_probabilities
        self.noise_level: float = noise_level
        self.S: int = sensor_positions.shape[0]  # Number of sensors

    @abstractmethod
    def __call__(self, data: torch.Tensor, seed: int = 0) -> torch.Tensor:
        pass


class Voronoi(SensorEmbedding):

    def __call__(self, data: torch.Tensor, seed: int = 0) -> torch.Tensor:
        N, T, C, H, W = data.shape
        assert (H, W) == (self.H, self.W)

        data = data.float()
        random.seed(seed)
        torch.manual_seed(seed)
        # Precompute dropout sensor selection for all samples and time frames
        n_dropout_sensors: torch.LongTensor = torch.multinomial(
            torch.tensor(self.dropout_probabilities, device='cuda'),
            num_samples=N * T, replacement=True,
        ).reshape(N, T)
        # Precompute dropout masks across all samples and frames (fast)
        masks: torch.Tensor = torch.ones((N, T, self.S), dtype=torch.bool, device='cuda')
        for i in range(N):
            for t in range(T):
                n_sensors_to_drop: int = n_dropout_sensors[i, t].item()
                if n_sensors_to_drop > 0:
                    dropout_indices = torch.randperm(self.S, device='cuda')[:n_sensors_to_drop]
                    masks[i, t, dropout_indices] = False

        del n_dropout_sensors   # manual garbage collection to save memory
        masks = masks.unsqueeze(-1).expand(N, T, self.S, H * W)
        assert masks.shape == (N, T, self.S, H * W)
        precomputed_distances_masked: torch.Tensor = self.precomputed_distances.unsqueeze(0).unsqueeze(0).expand(N, T, self.S, H * W)
        assert precomputed_distances_masked.shape == (N, T, self.S, H * W)
        # Set distances of dropped sensors to infinity
        precomputed_distances_masked = precomputed_distances_masked.masked_fill(mask=~masks, value=float('inf'))
        assert precomputed_distances_masked.shape == (N, T, self.S, H * W)
        nearest_sensor_per_position: torch.Tensor = torch.argmin(precomputed_distances_masked, dim=2)
        del precomputed_distances_masked
        assert nearest_sensor_per_position.shape == (N, T, H * W)
        
        assigned_sensor_per_position: torch.Tensor = self.sensor_positions[nearest_sensor_per_position.reshape(-1)].long()
        assert assigned_sensor_per_position.shape == (N * T * H * W, 2)
        assigned_sensor_per_position = assigned_sensor_per_position.reshape(N, T, H, W, 2)
        del nearest_sensor_per_position     # manual garbage collection to save memory
        
        h_indices: torch.LongTensor = assigned_sensor_per_position[..., 0]
        w_indices: torch.LongTensor = assigned_sensor_per_position[..., 1]
        output: torch.Tensor = torch.empty_like(data)
        noisy_data: torch.Tensor = data + torch.randn_like(data) * self.noise_level * data.abs()
        for i in range(N):
            for t in range(T):
                output[i, t] = noisy_data[i, t, :, h_indices[i, t], w_indices[i, t]]

        assert output.shape == data.shape == (N, T, C, H, W)
        return output

    @cached_property
    def precomputed_distances(self) -> torch.Tensor:
        # Create mesh grid for pixel positions
        grid_h, grid_w = torch.meshgrid(
            torch.arange(self.H, device='cuda'), torch.arange(self.W, device='cuda'), 
            indexing='ij',
        )
        grid_positions: torch.Tensor = torch.stack(tensors=[grid_h, grid_w], dim=2).reshape(self.H * self.W, 2).float()
        # Precompute distances for all sensors to all pixels
        differences: torch.Tensor = self.sensor_positions.unsqueeze(1) - grid_positions.unsqueeze(0)
        assert differences.shape == (self.S, self.H * self.W, 2)
        distance: torch.Tensor = (differences ** 2).sum(dim=2).sqrt()
        assert distance.shape == (self.S, self.H * self.W)
        return distance


class Mask(SensorEmbedding):

    def __call__(self, data: torch.Tensor, seed: int = 0) -> torch.Tensor:
        N, T, C, H, W = data.shape
        assert (H, W) == (self.H, self.W)

        data = data.float()
        # Control random seed
        random.seed(seed)
        torch.manual_seed(seed)
        noisy_data: torch.Tensor = data + torch.randn_like(data) * self.noise_level * data.abs()
        output: torch.Tensor = torch.empty_like(data, dtype=torch.float)
        for i in range(N):
            for t in range(T):
                n_dropout_sensors: int = random.choices(
                    population=range(0, self.n_max_dropout_sensors + 1, 1), weights=self.dropout_probabilities, k=1
                )[0]
                dropout_indices: torch.Tensor = torch.randperm(self.S)[:n_dropout_sensors]
                mask: torch.Tensor = torch.ones(self.S, dtype=torch.bool)
                mask[dropout_indices] = False
                remaining_sensor_positions: torch.Tensor = self.sensor_positions[mask].long()
                n_remaining_sensors: int = self.S - n_dropout_sensors
                assert remaining_sensor_positions.shape == (n_remaining_sensors, 2)
                h_indices: torch.Tensor = remaining_sensor_positions[:, 0]
                w_indices: torch.Tensor = remaining_sensor_positions[:, 1]
                output[i, t, :, h_indices, w_indices] = noisy_data[i, t, :, h_indices, w_indices]
        
        assert output.shape == data.shape == (N, T, C, H, W)
        return output


class Vector(SensorEmbedding):

    def __call__(self, data: torch.Tensor, seed: int = 0) -> torch.Tensor:
        N, T, C, H, W = data.shape
        assert (H, W) == (self.H, self.W)

        data = data.float()
        # Control random seed
        random.seed(seed)
        torch.manual_seed(seed)
        noisy_data: torch.Tensor = data + torch.randn_like(data) * self.noise_level * data.abs()
        output: torch.Tensor = torch.zeros((N, T, C, self.S), dtype=torch.float, device='cuda')
        for i in range(N):
            for t in range(T):
                n_dropout_sensors: int = random.choices(
                    population=range(0, self.n_max_dropout_sensors + 1, 1), weights=self.dropout_probabilities, k=1
                )[0]
                dropout_indices: torch.Tensor = torch.randperm(self.S)[:n_dropout_sensors]
                mask: torch.Tensor = torch.ones(self.S, dtype=torch.bool)
                mask[dropout_indices] = False
                remaining_sensor_positions: torch.Tensor = self.sensor_positions[mask].long()
                n_remaining_sensors: int = self.S - n_dropout_sensors
                assert remaining_sensor_positions.shape == (n_remaining_sensors, 2)
                h_indices: torch.Tensor = remaining_sensor_positions[:, 0]
                w_indices: torch.Tensor = remaining_sensor_positions[:, 1]
                output[i, t, :, mask] = noisy_data[i, t, :, h_indices, w_indices]
        
        assert output.shape == (N, T, C, self.S)
        return output
