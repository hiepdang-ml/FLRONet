import os
from abc import ABC, abstractmethod
from typing import List, Tuple
from functools import cache, cached_property

import numpy as np
import torch


class SensorGenerator(ABC):

    def __init__(
        self, 
        spatial_shape: Tuple[int, int], 
        n_sensors: int, 
    ) -> None:
        super().__init__()
        self.spatial_shape: Tuple[int, int] = spatial_shape
        self.n_sensors: int = n_sensors
        self.n_dims: int = len(spatial_shape)
        self.__seed: int = 0

    @abstractmethod
    def generate(self) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def abs2rel(self, abs_positions: torch.Tensor) -> torch.Tensor:
        rel_positions: torch.Tensor = torch.zeros_like(abs_positions, dtype=torch.float)
        rel_positions[:, 0] = abs_positions[:, 0].float() / self.spatial_shape[0]
        rel_positions[:, 1] = abs_positions[:, 1].float() / self.spatial_shape[1]
        return rel_positions

    @property
    def seed(self) -> int:
        return self.__seed
    
    @seed.setter
    def seed(self, value: int) -> None:
        self.__seed = value


class LHS(SensorGenerator):

    # implement
    @cache
    def generate(self) -> Tuple[torch.Tensor, torch.Tensor]:
        lhs_samples: np.ndarray = self._sampling()
        # absolute positions
        abs_sensor_positions: torch.Tensor = torch.zeros((self.n_sensors, self.n_dims), dtype=torch.int32)
        for sensor in range(self.n_sensors):
            for dim in range(self.n_dims):
                abs_sensor_positions[sensor, dim] = int(lhs_samples[sensor, dim] * self.spatial_shape[dim])

        # relative positions
        rel_sensor_positions: torch.Tensor = self.abs2rel(abs_sensor_positions)

        return rel_sensor_positions, sensor_positions

    def _sampling(self) -> np.ndarray:
        np.random.seed(self.seed)
        samples = np.zeros((self.n_sensors, self.n_dims))
        for dim in range(self.n_dims):
            segment_size = 1.0 / self.n_sensors
            segment_starts = np.arange(0, 1, segment_size)
            shuffled_segments: np.ndarray = np.random.permutation(segment_starts)
            for sensor in range(self.n_sensors):
                samples[sensor, dim] = np.random.uniform(
                    low=shuffled_segments[sensor], high=shuffled_segments[sensor] + segment_size
                )

        return samples
    

class AroundCylinder(SensorGenerator):

    # implement
    @cache
    def generate(
        self, 
        hw_meters: Tuple[float, float],
        center_hw_meters: Tuple[float, float],
        radius_meters: float, 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        assert self.n_dims == 2, 'AroundCylinder only works in 2D space'
        np.random.seed(self.seed)
        samples: np.ndarray = LHS(
            spatial_shape=(360,), n_sensors=self.n_sensors,
        )._sampling().flatten()
        samples = samples * 360
        # compute meters per pixel
        h_scale = hw_meters[0] / self.spatial_shape[0]
        w_scale = hw_meters[1] / self.spatial_shape[1]
        # compute radius of the cylinder
        radius_h_pixels = radius_meters / h_scale
        radius_w_pixels = radius_meters / w_scale
        # compute center of the cylinder
        center_h_pixels = center_hw_meters[0] / h_scale
        center_w_pixels = center_hw_meters[1] / w_scale

        # absolute positions
        abs_sensor_positions: torch.Tensor = torch.zeros((self.n_sensors, self.n_dims), dtype=torch.int32)
        abs_sensor_positions[:, 0] = torch.from_numpy(np.cos(np.deg2rad(samples)) * radius_h_pixels + center_h_pixels)
        abs_sensor_positions[:, 1] = torch.from_numpy(np.sin(np.deg2rad(samples)) * radius_w_pixels + center_w_pixels)

        # relative positions
        rel_sensor_positions: torch.Tensor = self.abs2rel(abs_sensor_positions)

        return rel_sensor_positions, abs_sensor_positions



if __name__ == '__main__':
    spatial_shape = (256, 512)
    # self = LHS(spatial_shape=(140, 240), n_sensors=32)
    # sensor_positions = self.generate()

    self = AroundCylinder(spatial_shape=spatial_shape, n_sensors=32)
    sensor_positions = self.generate(hw_meters=(0.14, 0.24), center_hw_meters=(0.08, 0.08), radius_meters=0.01)
    import matplotlib.pyplot as plt

    # Extract x and y coordinates
    x_coords = sensor_positions[:, 1].numpy()
    y_coords = sensor_positions[:, 0].numpy()

    plt.figure(figsize=(8, 4))
    plt.xlim(0, spatial_shape[1])
    plt.ylim(0, spatial_shape[0])
    plt.scatter(x_coords, y_coords, color='red', marker='o')
    plt.grid(True)
    plt.savefig('test.png')

    