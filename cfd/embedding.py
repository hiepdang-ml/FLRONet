from abc import ABC, abstractmethod
from typing import List
import random

import torch


class SensorEmbedding(ABC):

    @abstractmethod
    def __call__(self, data: torch.Tensor, sensor_positions: torch.LongTensor) -> torch.Tensor:
        pass


class Voronoi(SensorEmbedding):

    # Implement
    def __call__(self, data: torch.Tensor, sensor_positions: torch.LongTensor, dropout_probabilities: List[float] = []) -> torch.Tensor:
        assert data.ndim == 5 and sensor_positions.ndim == 2 and sensor_positions.shape[1] == 2, "Invalid input shapes"
        N, T, C, H, W = data.shape
        S: int = sensor_positions.shape[0]  # Number of sensors
        assert sum(dropout_probabilities) < 1, "Dropout probabilities must sum to less than 1"
        dropout_probabilities = [1 - sum(dropout_probabilities)] + dropout_probabilities

        data = data.float()
        # Create mesh grid for positions
        grid_h, grid_w = torch.meshgrid(
            torch.arange(H, device=data.device), 
            torch.arange(W, device=data.device), 
            indexing='ij'
        )
        positions: torch.Tensor = torch.stack(tensors=[grid_h, grid_w], dim=-1).reshape(-1, 2).float()
        assert positions.shape == (H * W, 2)
        output = torch.empty_like(data)
        for i in range(N):
            for t in range(T):
                # Randomly drop sensors for each sample and time frame
                n_dropout_sensors: int = random.choices(population=range(S), weights=dropout_probabilities, k=1)[0]
                dropout_indices: torch.Tensor = torch.randperm(S)[:n_dropout_sensors]
                mask: torch.Tensor = torch.ones(S, dtype=torch.bool)
                mask[dropout_indices] = False
                remaining_sensor_positions = sensor_positions[mask]
                n_remaining_sensors: int = remaining_sensor_positions.shape[0]
                # Compute distance of each pixel to the remaining sensors
                distances: torch.Tensor = torch.norm(remaining_sensor_positions.unsqueeze(1) - positions, dim=-1)
                assert distances.shape == (n_remaining_sensors, H * W)
                # Voronoi Nearest Neighbor Interpolation
                nearest_sensor_per_position: torch.Tensor = torch.argmin(distances, dim=0)
                assert nearest_sensor_per_position.shape == (H * W,)
                assigned_sensor_per_position: torch.Tensor = remaining_sensor_positions[nearest_sensor_per_position]
                assert assigned_sensor_per_position.shape == (H * W, 2)

                h_indices: torch.Tensor = assigned_sensor_per_position[:, 0].long()
                w_indices: torch.Tensor = assigned_sensor_per_position[:, 1].long()
                output[i, t, :, :, :] = data[i, t, :, h_indices, w_indices].reshape(C, H, W)

        assert output.shape == data.shape == (N, T, C, H, W)
        return output


class Mask(SensorEmbedding):

    # Implement
    def __call__(self, data: torch.Tensor, sensor_positions: torch.LongTensor, dropout_probabilities: List[float]) -> torch.Tensor:
        assert data.ndim == 5 and sensor_positions.ndim == 2 and sensor_positions.shape[1] == 2, "Invalid input shapes"
        N, T, C, H, W = data.shape
        S: int = sensor_positions.shape[0]  # Number of sensors
        assert sum(dropout_probabilities) < 1, "Dropout probabilities must sum to less than 1"
        dropout_probabilities = [1 - sum(dropout_probabilities)] + dropout_probabilities

        output: torch.Tensor = torch.zeros_like(data, dtype=torch.float)
        for i in range(N):
            for t in range(T):
                # Randomly drop sensors for each sample and time frame
                n_dropout_sensors: int = random.choices(population=range(S), weights=dropout_probabilities, k=1)[0]
                dropout_indices: torch.Tensor = torch.randperm(S)[:n_dropout_sensors]
                mask: torch.Tensor = torch.ones(S, dtype=torch.bool)
                mask[dropout_indices] = False
                remaining_sensor_positions = sensor_positions[mask]
                h_indices: torch.Tensor = remaining_sensor_positions[:, 0]
                w_indices: torch.Tensor = remaining_sensor_positions[:, 1]
                output[i, t, :, h_indices, w_indices] = data[i, t, :, h_indices, w_indices]

        assert output.shape == data.shape == (N, T, C, H, W)
        return output


if __name__ == '__main__':
    x = torch.arange(50, dtype=torch.float).reshape(1, 1, 1, 5, 10)
    # self = Voronoi()
    self = Voronoi()
    points = torch.tensor([[1, 4], [2, 6], [3, 7]])
    a = self(x, points)
    print(a)

    self = Mask()
    b = self(x, points)
    print(b)


