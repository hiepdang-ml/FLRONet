import os
from abc import ABC, abstractmethod
from typing import List, Tuple
from functools import cache

import numpy as np
import torch


class SensorGenerator(ABC):

    def __init__(
        self, 
        n_sensors: int, 
    ) -> None:
        super().__init__()
        self.n_sensors: int = n_sensors
        self.__seed: int = 0
        self.__resolution: Tuple[int, int] | None = None

    @abstractmethod
    def __call__(self) -> torch.Tensor:
        pass

    @property
    def seed(self) -> int:
        return self.__seed
    
    @seed.setter
    def seed(self, value: int) -> None:
        self.__seed = value

    @property
    def resolution(self) -> Tuple[int, int]:
        return self.__resolution
    
    @resolution.setter
    def resolution(self, value: Tuple[int, int]) -> None:
        self.__resolution = value


class LHS(SensorGenerator):

    # implement
    @cache
    def __call__(self) -> torch.Tensor:
        assert self.resolution is not None, 'self.resolution must be set before calling a SensorGenerator'
        lhs_samples: np.ndarray = self._sampling()
        # absolute positions
        sensor_positions = torch.from_numpy(lhs_samples) * torch.tensor(data=self.resolution)
        return sensor_positions.int()

    def _sampling(self) -> np.ndarray:
        np.random.seed(self.seed)
        samples = np.zeros((self.n_sensors, len(self.resolution)))
        for dim in range(len(self.resolution)):
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
        assert self.resolution is not None, 'self.resolution must be set before calling a SensorGenerator'
        assert len(self.resolution) == 2, 'AroundCylinder only works in 2D space'
        np.random.seed(self.seed)
        lhs = LHS(n_sensors=self.n_sensors); lhs.resolution = (360,)
        samples: np.ndarray = lhs._sampling().flatten()
        samples = samples * 360
        # compute meters per pixel
        h_scale: float = hw_meters[0] / self.resolution[0]
        w_scale: float = hw_meters[1] / self.resolution[1]
        # compute radius of the cylinder
        radius_w_pixels: float = radius_meters / w_scale
        radius_h_pixels: float = radius_meters / h_scale
        # compute center of the cylinder
        center_h_pixels: float = center_hw_meters[0] / h_scale
        center_w_pixels: float = center_hw_meters[1] / w_scale
        # compute sensor positions
        sensor_positions: torch.Tensor = torch.zeros((self.n_sensors, len(self.resolution)), dtype=torch.int32)
        sensor_positions[:, 0] = torch.from_numpy(np.cos(np.deg2rad(samples)) * radius_h_pixels + center_h_pixels)
        sensor_positions[:, 1] = torch.from_numpy(np.sin(np.deg2rad(samples)) * radius_w_pixels + center_w_pixels)
        return sensor_positions



if __name__ == '__main__':
    resolution = (140, 240)
    lhs = LHS(n_sensors=32)
    lhs.resolution = resolution
    ac = AroundCylinder(n_sensors=32)
    ac.resolution = resolution
    a = lhs()
    b = ac(hw_meters=(0.14, 0.24), center_hw_meters=(0.08, 0.08), radius_meters=0.01)
    import matplotlib.pyplot as plt

    for name, sensor_positions in zip(('lhs', 'ac'), (a, b)):
        # Extract x and y coordinates
        x_coords = sensor_positions[:, 1].numpy()
        y_coords = sensor_positions[:, 0].numpy()

        plt.figure(figsize=(8, 4))
        plt.xlim(0, resolution[1])
        plt.ylim(0, resolution[0])
        plt.scatter(x_coords, y_coords, color='red', marker='o')
        plt.grid(True)
        plt.savefig(f'{name}.png')

    