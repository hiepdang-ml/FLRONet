from abc import ABC, abstractmethod
import torch


class SensorEmbedding(ABC):

    @abstractmethod
    def __call__(self, data: torch.Tensor, sensor_positions: torch.LongTensor) -> torch.Tensor:
        pass


class Voronoi(SensorEmbedding):

    def __init__(self, weighted: bool = False):
        super().__init__()
        self.weighted: bool = weighted

    # implement
    def __call__(self, data: torch.Tensor, sensor_positions: torch.LongTensor) -> torch.Tensor:
        assert data.ndim == 5 and sensor_positions.ndim == 2 and sensor_positions.shape[1] == 2
        N, T, C, H, W = data.shape
        S: int = sensor_positions.shape[0]  # number of sensors

        data = data.float()
        grid_h, grid_w = torch.meshgrid(
            torch.arange(H, device=data.device), 
            torch.arange(W, device=data.device), 
            indexing='ij'
        )
        positions: torch.Tensor = torch.stack(tensors=[grid_h, grid_w], dim=-1).reshape(-1, 2).float()
        assert positions.shape == (H * W, 2)
        distances: torch.Tensor = torch.norm(sensor_positions.unsqueeze(1) - positions, dim=-1)
        assert distances.shape == (S, H * W)
        # Voronoi Variant: Weighted average interpolation
        if self.weighted:
            h_indices: torch.Tensor = sensor_positions[:, 0].long()
            w_indices: torch.Tensor = sensor_positions[:, 1].long()
            sensor_values: torch.Tensor = data[:, :, :, h_indices, w_indices]
            assert sensor_values.shape == (N, T, C, S)
            # inverse distance weighting
            epsilon: float = 1e-8
            inverse_distances: torch.Tensor = 1.0 / (distances + epsilon)
            weights: torch.Tensor = inverse_distances / inverse_distances.sum(dim=0, keepdim=True)
            assert weights.shape == (S, H * W)
            output: torch.Tensor = torch.sum(
                # (1, 1, 1, S, H * W) * (N, T, C, S, 1) = (N, T, C, S, H * W)
                weights.unsqueeze(0).unsqueeze(0).unsqueeze(0) * sensor_values.unsqueeze(-1), 
                dim=3,
            ).reshape(N, T, C, H, W)
        # Original Voronoi: Nearest Neighbor Interpolation
        else:
            closest_sensor_per_position: torch.Tensor = torch.argmin(distances, dim=0)
            assert closest_sensor_per_position.shape == (H * W,)
            assigned_sensor_per_position: torch.Tensor = sensor_positions[closest_sensor_per_position]
            assert assigned_sensor_per_position.shape == (H * W, 2)

            h_indices: torch.Tensor = assigned_sensor_per_position[:, 0].long()
            w_indices: torch.Tensor = assigned_sensor_per_position[:, 1].long()
            output: torch.Tensor = data[:, :, :, h_indices, w_indices].reshape(N, T, C, H, W)

        assert output.shape == data.shape == (N, T, C, H, W)
        return output


class Mask(SensorEmbedding):

    # implement
    def __call__(self, data: torch.Tensor, sensor_positions: torch.LongTensor) -> torch.Tensor:
        assert data.ndim == 5 and sensor_positions.ndim == 2 and sensor_positions.shape[1] == 2
        N, T, C, H, W = data.shape
        S: int = sensor_positions.shape[0]  # number of sensors
        output: torch.Tensor = torch.zeros_like(data, dtype=torch.float)
        sensor_x: torch.Tensor = sensor_positions[:, 0]
        sensor_y: torch.Tensor = sensor_positions[:, 1]
        output[:, :, :, sensor_x, sensor_y] = data[:, :, :, sensor_x, sensor_y]
        assert output.shape == data.shape == (N, T, C, H, W)
        return output



if __name__ == '__main__':
    x = torch.arange(50, dtype=torch.float).reshape(1, 1, 1, 5, 10)
    # self = Voronoi(weighted=False)
    self = Voronoi(weighted=True)
    points = torch.tensor([[1, 4], [2, 6], [3, 7]])
    a = self(x, points)
    print(a)

    self = Mask()
    b = self(x, points)
    print(b)


