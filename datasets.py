import os
from typing import List, Tuple, Optional
import shutil
import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from sensors import LHS, AroundCylinder
from interpolation import Voronoi, Mask


class CFDDataset(Dataset):

    def __init__(
        self, 
        root: str, 
        init_sensor_timeframe_indices: List[int],
        n_fullstate_timeframes_per_chunk: int,
        resolution: Tuple[int, int] | None,
        sensor_generator: LHS | AroundCylinder, 
        embedding_generator: Voronoi | Mask,
        seed: int,
    ) -> None:
        
        super().__init__()
        self.case_directories: List[str] = sorted(
            [os.path.join(root, casedir) for casedir in os.listdir(root)]
        )

        self.root: str = root
        self.init_sensor_timeframe_indices: List[int] = init_sensor_timeframe_indices
        self.n_fullstate_timeframes_per_chunk: int = n_fullstate_timeframes_per_chunk
        self.resolution: Tuple[int, int] | None = resolution
        self.sensor_generator: LHS | AroundCylinder = sensor_generator
        self.embedding_generator: Voronoi | Mask = embedding_generator
        self.seed: int = seed

        self.n_sensor_timeframes_per_chunk: int = len(init_sensor_timeframe_indices)
        self.total_timeframes_per_case: int = np.load(os.path.join(self.case_directories[0], 'u.npy')).shape[0]

        self.sensor_timeframes_dest: str = os.path.join('tensors', 'sensor_timeframes')
        self.sensor_values_dest: str = os.path.join('tensors', 'sensor_values')
        self.fullstate_timeframes_dest: str = os.path.join('tensors', 'fullstate_timeframes')
        self.fullstate_values_dest: str = os.path.join('tensors', 'fullstate_values')
        
        self.sensor_timeframe_indices = self.__prepare_sensor_timeframe_indices()
        self.__write2disk()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        suffix: str = str(idx).zfill(6)
        sensor_timeframe_tensor: torch.Tensor = torch.load(os.path.join(self.sensor_timeframes_dest, f'ST{suffix}.pt'))
        sensor_tensor: torch.Tensor = torch.load(os.path.join(self.sensor_values_dest, f'SV{suffix}.pt'))
        fullstate_timeframe_tensor: torch.Tensor = torch.load(os.path.join(self.fullstate_timeframes_dest, f'FT{suffix}.pt'))
        fullstate_tensor: torch.Tensor = torch.load(os.path.join(self.fullstate_values_dest, f'FV{suffix}.pt'))
        return sensor_timeframe_tensor, sensor_tensor, fullstate_timeframe_tensor, fullstate_tensor
    
    def __len__(self) -> int:
        return len([f for f in os.listdir(self.fullstate_values_dest) if f.endswith('.pt')])

    def __write2disk(self) -> None:

        # prepare dest directories
        shutil.rmtree('tensors')
        os.makedirs(name=self.sensor_timeframes_dest, exist_ok=True)
        os.makedirs(name=self.sensor_values_dest, exist_ok=True)
        os.makedirs(name=self.fullstate_timeframes_dest, exist_ok=True)
        os.makedirs(name=self.fullstate_values_dest, exist_ok=True)

        for caseid, casedir in enumerate(self.case_directories[:2]):
            data: torch.Tensor = torch.stack(
                tensors=[
                    torch.from_numpy(np.load(os.path.join(casedir, 'u.npy'))),
                    torch.from_numpy(np.load(os.path.join(casedir, 'v.npy')))
                ],
                dim=1
            )
            # sensor data
            sensor_data: torch.Tensor = data[self.sensor_timeframe_indices]
            n_chunks: int = sensor_data.shape[0]
            H, W = sensor_data.shape[-2:]
            assert sensor_data.shape == (n_chunks, self.n_sensor_timeframes_per_chunk, 2, H, W)
            # fullstate data
            fullstate_timeframe_indices: torch.Tensor = self.__prepare_fullstate_timeframe_indices(seed=self.seed + caseid)
            fullstate_data: torch.Tensor = data[fullstate_timeframe_indices]
            assert fullstate_data.shape == (n_chunks, self.n_fullstate_timeframes_per_chunk, 2, H, W)

            assert sensor_data.shape == (n_chunks, self.n_sensor_timeframes_per_chunk, 2, H, W)
            assert fullstate_data.shape == (n_chunks, self.n_fullstate_timeframes_per_chunk, 2, H, W)

            # adjust resolution (original resolution is 64 x 64, which is not proportional to 0.14m x 0.24m)
            if self.resolution is not None:
                H, W = self.resolution
                # resize sensor
                sensor_data = F.interpolate(input=sensor_data.flatten(0, 1), size=self.resolution, mode='bicubic')
                sensor_data = sensor_data.reshape(n_chunks, self.n_sensor_timeframes_per_chunk, 2, H, W)
                # resize fullstate
                fullstate_data = F.interpolate(input=fullstate_data.flatten(0, 1), size=self.resolution, mode='bicubic')
                fullstate_data = fullstate_data.reshape(n_chunks, self.n_fullstate_timeframes_per_chunk, 2, H, W)
                fullstate_data = fullstate_data.float()

            # prepare sensor positions
            if isinstance(self.sensor_generator, LHS):
                sensor_positions = self.sensor_generator()
            else:
                sensor_positions = self.sensor_generator(
                    hw_meters=(0.14, 0.24), center_hw_meters=(0.08, 0.08), radius_meters=0.01
                )

            # prepare embeddings for sensor data
            sensor_data = self.embedding_generator(data=sensor_data, sensor_positions=sensor_positions)
            sensor_data = sensor_data.float()

            assert sensor_data.shape == (n_chunks, self.n_sensor_timeframes_per_chunk, 2, H, W)
            assert fullstate_data.shape == (n_chunks, self.n_fullstate_timeframes_per_chunk, 2, H, W)
            assert fullstate_timeframe_indices.shape == (n_chunks, self.n_fullstate_timeframes_per_chunk)

            for idx in tqdm.tqdm(range(n_chunks), desc=f'Case {caseid + 1}'):
                true_idx: int = idx + caseid * n_chunks
                suffix = str(true_idx).zfill(6)
                # save sensor timeframes (constant)
                torch.save(obj=self.sensor_timeframe_indices[idx], f=os.path.join(self.sensor_timeframes_dest, f'ST{suffix}.pt'))
                # save sensor value, fullstate timeframes, fullstate data (dynamic)
                torch.save(obj=sensor_data[idx].clone(), f=os.path.join(self.sensor_values_dest, f'SV{suffix}.pt'))
                torch.save(obj=fullstate_timeframe_indices[idx].clone(), f=os.path.join(self.fullstate_timeframes_dest, f'FT{suffix}.pt'))
                torch.save(obj=fullstate_data[idx].clone(), f=os.path.join(self.fullstate_values_dest, f'FV{suffix}.pt'))
        
    def __prepare_sensor_timeframe_indices(self) -> torch.LongTensor:
        # compute number of steps to reach n_timeframes
        steps = self.total_timeframes_per_case - max(self.init_sensor_timeframe_indices)
        # prepare sensor timeframe indices (fixed)
        sensor_timeframe_indices: torch.Tensor = torch.tensor(self.init_sensor_timeframe_indices) + torch.arange(steps).unsqueeze(1)
        assert sensor_timeframe_indices.shape == (steps, len(self.init_sensor_timeframe_indices))
        return sensor_timeframe_indices

    def __prepare_fullstate_timeframe_indices(self, seed: int) -> torch.Tensor:
        # compute number of steps to reach n_timeframes
        steps = self.total_timeframes_per_case - max(self.init_sensor_timeframe_indices)
        # prepare fullstate timeframe indices (stochastic)
        torch.random.manual_seed(seed=seed)
        random_init_timeframe_indices: torch.Tensor = torch.randperm(
            n=max(self.init_sensor_timeframe_indices)
        )[:self.n_fullstate_timeframes_per_chunk].sort()[0]
        fullstate_timeframe_indices: torch.Tensor = random_init_timeframe_indices + torch.arange(steps).unsqueeze(1)
        assert fullstate_timeframe_indices.shape == (steps, self.n_fullstate_timeframes_per_chunk)
        return fullstate_timeframe_indices


if __name__ == '__main__':
    # sensor_generator = LHS(spatial_shape=(64, 64), n_sensors=32)
    sensor_generator = AroundCylinder(spatial_shape=(64, 64), n_sensors=32)
    # embedding_generator = Mask()
    embedding_generator = Voronoi(weighted=False)

    self = CFDDataset(
        root='./bc', 
        init_sensor_timeframe_indices=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        n_fullstate_timeframes_per_chunk=10,
        resolution=(140, 240),
        sensor_generator=sensor_generator, 
        embedding_generator=embedding_generator,
        seed=1,
    )
    sensor_timeframe_tensor, sensor_tensor, fullstate_timeframe_tensor, fullstate_tensor = self[500]
    print(sensor_timeframe_tensor)
    print(sensor_tensor)
    print(fullstate_timeframe_tensor)
    print(fullstate_tensor)






