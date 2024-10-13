import os
from abc import ABC, abstractmethod
from typing import List, Tuple
from functools import cache, cached_property

import numpy as np
import torch


class SensorGenerator(ABC):

    def __init__(
        self, 
        resolution: Tuple[int, int], 
        n_sensors: int, 
    ) -> None:
        super().__init__()
        self.resolution: Tuple[int, int] = resolution
        self.n_sensors: int = n_sensors
        self.n_dims: int = len(resolution)
        self.__seed: int = 0

    @abstractmethod
    def __call__(self) -> torch.Tensor:
        pass

    @property
    def seed(self) -> int:
        return self.__seed
    
    @seed.setter
    def seed(self, value: int) -> None:
        self.__seed = value


class LHS(SensorGenerator):

    # implement
    @cache
    def __call__(self) -> torch.Tensor:
        lhs_samples: np.ndarray = self._sampling()
        # absolute positions
        sensor_positions: torch.Tensor = torch.zeros((self.n_sensors, self.n_dims), dtype=torch.int32)
        for sensor in range(self.n_sensors):
            for dim in range(self.n_dims):
                sensor_positions[sensor, dim] = int(lhs_samples[sensor, dim] * self.resolution[dim])

        return sensor_positions

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
    def __call__(
        self, 
        hw_meters: Tuple[float, float],
        center_hw_meters: Tuple[float, float],
        radius_meters: float, 
    ) -> torch.Tensor:
        
        assert self.n_dims == 2, 'AroundCylinder only works in 2D space'
        np.random.seed(self.seed)
        samples: np.ndarray = LHS(
            resolution=(360,), n_sensors=self.n_sensors,
        )._sampling().flatten()
        samples = samples * 360
        # compute meters per pixel
        h_scale = hw_meters[0] / self.resolution[0]
        w_scale = hw_meters[1] / self.resolution[1]
        # compute radius of the cylinder
        radius_h_pixels = radius_meters / h_scale
        radius_w_pixels = radius_meters / w_scale
        # compute center of the cylinder
        center_h_pixels = center_hw_meters[0] / h_scale
        center_w_pixels = center_hw_meters[1] / w_scale

        print(h_scale, w_scale)
        print(center_h_pixels, center_w_pixels)

        sensor_positions: torch.Tensor = torch.zeros((self.n_sensors, self.n_dims), dtype=torch.int32)
        sensor_positions[:, 0] = torch.from_numpy(np.cos(np.deg2rad(samples)) * radius_h_pixels + center_h_pixels)
        sensor_positions[:, 1] = torch.from_numpy(np.sin(np.deg2rad(samples)) * radius_w_pixels + center_w_pixels)

        return sensor_positions



if __name__ == '__main__':
    resolution = (256, 512)
    # self = LHS(resolution=(140, 240), n_sensors=32)
    # sensor_positions = self.generate()

    self = AroundCylinder(resolution=resolution, n_sensors=32)
    sensor_positions = self(hw_meters=(0.14, 0.24), center_hw_meters=(0.08, 0.08), radius_meters=0.01)
    import matplotlib.pyplot as plt

    # Extract x and y coordinates
    x_coords = sensor_positions[:, 1].numpy()
    y_coords = sensor_positions[:, 0].numpy()

    plt.figure(figsize=(8, 4))
    plt.xlim(0, resolution[1])
    plt.ylim(0, resolution[0])
    plt.scatter(x_coords, y_coords, color='red', marker='o')
    plt.grid(True)
    plt.savefig('test.png')

    