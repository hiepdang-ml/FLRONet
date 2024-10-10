from abc import ABC, abstractmethod
import torch


class EmbeddingGenerator(ABC):

    @abstractmethod
    def __call__(self, data: torch.Tensor, sensor_positions: torch.LongTensor) -> torch.Tensor:
        pass


class Voronoi(EmbeddingGenerator):

    def __init__(self, weighted: bool = False):
        super().__init__()
        self.weighted: bool = weighted

    # implement
    def __call__(self, data: torch.Tensor, sensor_positions: torch.LongTensor) -> torch.Tensor:
        assert data.ndim == 5 and sensor_positions.ndim == 2 and sensor_positions.shape[1] == 2
        N, T, C, H, W = data.shape
        S: int = sensor_positions.shape[0]  # number of sensors

        grid_h, grid_w = torch.meshgrid(
            torch.arange(H, device=data.device), 
            torch.arange(W, device=data.device), 
            indexing='ij'
        )
        positions: torch.Tensor = torch.stack(tensors=[grid_h, grid_w], dim=-1).reshape(-1, 2).float()
        assert positions.shape == (H * W, 2)

        distances: torch.Tensor = torch.norm(sensor_positions.unsqueeze(1) - positions, dim=-1)
        assert distances.shape == (S, H * W)
        closest_sensor_per_position: torch.Tensor = torch.argmin(distances, dim=0)
        assert closest_sensor_per_position.shape == (H * W,)
        assigned_sensor_per_position: torch.Tensor = sensor_positions[closest_sensor_per_position]
        assert assigned_sensor_per_position.shape == (H * W, 2)

        output: torch.Tensor = torch.zeros_like(data, dtype=torch.float)

        # Voronoi Variant: Weighted average interpolation
        if self.weighted:
            epsilon: float = 1e-8
            inverse_distances: torch.Tensor = 1.0 / (distances + epsilon)
            # inverse distance weighting
            weights: torch.Tensor = inverse_distances / inverse_distances.sum(dim=0, keepdim=True)  # (S, H * W)

            for n in range(N):
                for t in range(T):
                    for c in range(C):
                        sensor_values: torch.Tensor = data[n, t, c, sensor_positions[:, 0], sensor_positions[:, 1]]
                        assert sensor_values.shape == (S,)
                        interpolated_values: torch.Tensor = torch.sum(weights * sensor_values.unsqueeze(1), dim=0).reshape(H, W)
                        output[n, t, c] = interpolated_values
        
        # Original Voronoi: Nearest Neighbor Interpolation
        else:
            for n in range(N):
                for t in range(T):
                    for c in range(C):
                        interpolated_values: torch.Tensor = data[
                            n, t, c, assigned_sensor_per_position[:, 0], assigned_sensor_per_position[:, 1]
                        ].reshape(H, W)
                        output[n, t, c] = interpolated_values

        assert output.shape == data.shape == (N, T, C, H, W)
        return output


class Mask(EmbeddingGenerator):

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
    self = Voronoi(weighted=False)
    points = torch.tensor([[1, 4], [2, 6], [3, 7]])
    a = self(x, points)
    print(a)

    self = Mask()
    b = self(x, points)
    print(b)


