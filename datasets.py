import os
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from sensors import LHS, AroundCylinder


class CFDDataset(Dataset):

    # TODO: implement interpolation to arbitrary shape
    def __init__(self, root: str, sensor_generator: LHS | AroundCylinder) -> None:
        super().__init__()
        self.root: str = root
        self.sensor_generator: LHS | AroundCylinder = sensor_generator
        self.case_directories: List[str] = sorted([os.path.join(root, casedir) for casedir in os.listdir(root)])
        self.full_state, self.sensor_value = self.__load2ram()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.full_state[idx], self.sensor_value[idx]
    
    def __len__(self) -> int:
        return self.full_state.shape[0]

    def __load2ram(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # prepare fullstate
        full_states: List[torch.Tensor] = [
            torch.stack(
                tensors=[
                    torch.from_numpy(np.load(os.path.join(casedir, 'u.npy'))),
                    torch.from_numpy(np.load(os.path.join(casedir, 'v.npy')))
                ],
                dim=1
            )
            for casedir in self.case_directories
        ]
        full_state: torch.Tensor = torch.cat(tensors=full_states, dim=0) # (N, 2, H, W)
        n_samples: int = full_state.shape[0]; H, W = full_state.shape[-2:]
        assert full_state.shape[1] == 2

        # prepare sensor values
        if isinstance(self.sensor_generator, LHS):
            self.sensor_positions: torch.Tensor = self.sensor_generator.generate()
        else:
            self.sensor_positions: torch.Tensor = self.sensor_generator.generate(
                hw_meters=(0.14, 0.24), center_hw_meters=(0.08, 0.08), radius_meters=0.01
            )

        h_indices: torch.Tensor = self.sensor_positions[:, 0].unsqueeze(dim=0)
        w_indices: torch.Tensor = self.sensor_positions[:, 1].unsqueeze(dim=0)
        assert h_indices.shape == w_indices.shape == (1, self.sensor_generator.n_sensors)
        # advance indexing
        sensor_value: torch.Tensor = full_state[
            torch.arange(n_samples).unsqueeze(dim=1), :, h_indices, w_indices
        ]
        sensor_value = sensor_value.permute(0, 2, 1)
        assert sensor_value.shape == (n_samples, 2, self.sensor_generator.n_sensors)

        return full_state, sensor_value
    
    

if __name__ == '__main__':
    sensor_generator = LHS(spatial_shape=(64, 64), n_sensors=32)
    sensor_generator = AroundCylinder(spatial_shape=(64, 64), n_sensors=32)
    self = CFDDataset(root='./bc', sensor_generator=sensor_generator)
    full_state, sensor_value = self[500]
    full_state
    sensor_value






