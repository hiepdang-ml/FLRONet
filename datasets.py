import os
from typing import List, Tuple, Optional
import shutil
import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from sensors import LHS, AroundCylinder
from embeddings import Voronoi, Mask


class CFDDataset(Dataset):

    def __init__(
        self, 
        root: str, 
        init_sensor_timeframe_indices: List[int],
        n_fullstate_timeframes_per_chunk: int,
        n_samplings_per_chunk: int,
        resolution: Tuple[int, int],
        sensor_generator: LHS | AroundCylinder, 
        embedding_generator: Voronoi | Mask,
        seed: int,
    ) -> None:
        
        super().__init__()
        self.case_directories: List[str] = sorted(
            [os.path.join(root, case_dir) for case_dir in os.listdir(root)]
        )

        self.root: str = root
        self.init_sensor_timeframe_indices: List[int] = init_sensor_timeframe_indices
        self.n_fullstate_timeframes_per_chunk: int = n_fullstate_timeframes_per_chunk
        self.n_samplings_per_chunk: int = n_samplings_per_chunk
        self.resolution: Tuple[int, int] = resolution
        self.sensor_generator: LHS | AroundCylinder = sensor_generator
        self.sensor_generator.seed = seed
        self.embedding_generator: Voronoi | Mask = embedding_generator
        self.seed: int = seed

        self.H, self.W = self.resolution
        self.n_sensor_timeframes_per_chunk: int = len(init_sensor_timeframe_indices)
        self.total_timeframes_per_case: int = np.load(os.path.join(self.case_directories[0], 'u.npy')).shape[0]

        self.sensor_positions_dest: str = os.path.join('tensors', 'sensor_positions')
        self.sensor_timeframes_dest: str = os.path.join('tensors', 'sensor_timeframes')
        self.sensor_values_dest: str = os.path.join('tensors', 'sensor_values')
        self.fullstate_timeframes_dest: str = os.path.join('tensors', 'fullstate_timeframes')
        self.fullstate_values_dest: str = os.path.join('tensors', 'fullstate_values')
        
        self.sensor_timeframe_indices = self.__prepare_sensor_timeframe_indices()
        self.__write2disk()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        suffix: str = str(idx).zfill(6)
        sensor_timeframe_tensor: torch.Tensor = torch.load(os.path.join(self.sensor_timeframes_dest, f'ST{suffix}.pt'), weights_only=True)
        sensor_tensor: torch.Tensor = torch.load(os.path.join(self.sensor_values_dest, f'SV{suffix}.pt'), weights_only=True)
        fullstate_timeframe_tensor: torch.Tensor = torch.load(os.path.join(self.fullstate_timeframes_dest, f'FT{suffix}.pt'), weights_only=True)
        fullstate_tensor: torch.Tensor = torch.load(os.path.join(self.fullstate_values_dest, f'FV{suffix}.pt'), weights_only=True)
        return sensor_timeframe_tensor, sensor_tensor, fullstate_timeframe_tensor, fullstate_tensor
    
    def __len__(self) -> int:
        return len([f for f in os.listdir(self.fullstate_values_dest) if f.endswith('.pt')])

    def __write2disk(self) -> None:

        # prepare dest directories
        shutil.rmtree('tensors')
        os.makedirs(name=self.sensor_positions_dest, exist_ok=True)
        os.makedirs(name=self.sensor_timeframes_dest, exist_ok=True)
        os.makedirs(name=self.sensor_values_dest, exist_ok=True)
        os.makedirs(name=self.fullstate_timeframes_dest, exist_ok=True)
        os.makedirs(name=self.fullstate_values_dest, exist_ok=True)

        # prepare sensor positions
        if isinstance(self.sensor_generator, LHS):
            self.sensor_positions = self.sensor_generator()
        else:
            self.sensor_positions = self.sensor_generator(
                hw_meters=(0.14, 0.24), center_hw_meters=(0.07, 0.065), radius_meters=0.01,
            )

        assert self.sensor_positions.shape == (self.sensor_generator.n_sensors, 2)
        # save sensor for reference (it's not used in model)
        torch.save(obj=self.sensor_positions, f=os.path.join(self.sensor_positions_dest, 'pos.pt'))

        # for case_id, case_dir in enumerate(self.case_directories):
        for case_id, case_dir in enumerate(self.case_directories[:1]):
            data: torch.Tensor = torch.stack(
                tensors=[
                    torch.from_numpy(np.load(os.path.join(case_dir, 'u.npy'))),
                    torch.from_numpy(np.load(os.path.join(case_dir, 'v.npy')))
                ],
                dim=1
            )
            # sensor data
            sensor_data: torch.Tensor = data[self.sensor_timeframe_indices]
            n_chunks: int = sensor_data.shape[0]

            # resize sensor frames (original resolution is 64 x 64, which is not proportional to 0.14m x 0.24m)
            sensor_data = F.interpolate(input=sensor_data.flatten(0, 1), size=self.resolution, mode='bicubic')
            sensor_data = sensor_data.reshape(n_chunks, self.n_sensor_timeframes_per_chunk, 2, self.H, self.W)
            
            # compute sensor data for entire space
            sensor_data = self.embedding_generator(data=sensor_data, sensor_positions=self.sensor_positions)
            assert sensor_data.shape == (n_chunks, self.n_sensor_timeframes_per_chunk, 2, self.H, self.W)

            for sampling_id in range(self.n_samplings_per_chunk):
                # fullstate data
                fullstate_timeframe_indices: torch.Tensor = self.__prepare_fullstate_timeframe_indices(seed=self.seed + case_id + sampling_id)
                fullstate_data: torch.Tensor = data[fullstate_timeframe_indices]
                # resize fullstate frames
                fullstate_data = F.interpolate(input=fullstate_data.flatten(0, 1), size=self.resolution, mode='bicubic')
                fullstate_data = fullstate_data.reshape(n_chunks, self.n_fullstate_timeframes_per_chunk, 2, self.H, self.W)

                assert fullstate_data.shape == (n_chunks, self.n_fullstate_timeframes_per_chunk, 2, self.H, self.W)
                assert fullstate_timeframe_indices.shape == (n_chunks, self.n_fullstate_timeframes_per_chunk)

                for idx in tqdm.tqdm(range(n_chunks), desc=f'Case {case_id + 1} | Sampling {sampling_id + 1}: '):
                    true_idx: int = idx + sampling_id * n_chunks + case_id * n_chunks * self.n_samplings_per_chunk
                    suffix = str(true_idx).zfill(6)
                    # save sensor timeframes, sensor value (dynamic to chunks, but constant to samplings)
                    torch.save(obj=self.sensor_timeframe_indices[idx].clone(), f=os.path.join(self.sensor_timeframes_dest, f'ST{suffix}.pt'))
                    torch.save(obj=sensor_data[idx].clone(), f=os.path.join(self.sensor_values_dest, f'SV{suffix}.pt'))
                    # save sensor value, fullstate timeframes, fullstate data (fully dynamic)
                    torch.save(obj=fullstate_timeframe_indices[idx].clone(), f=os.path.join(self.fullstate_timeframes_dest, f'FT{suffix}.pt'))
                    torch.save(obj=fullstate_data[idx].clone(), f=os.path.join(self.fullstate_values_dest, f'FV{suffix}.pt'))
            
    def __prepare_sensor_timeframe_indices(self) -> torch.LongTensor:
        # compute number of steps to reach n_timeframes (also the number of chunks)
        n_chunks = self.total_timeframes_per_case - max(self.init_sensor_timeframe_indices)
        # prepare sensor timeframe indices (fixed)
        sensor_timeframe_indices: torch.Tensor = torch.tensor(self.init_sensor_timeframe_indices) + torch.arange(n_chunks).unsqueeze(1)
        assert sensor_timeframe_indices.shape == (n_chunks, len(self.init_sensor_timeframe_indices))
        return sensor_timeframe_indices

    def __prepare_fullstate_timeframe_indices(self, seed: int) -> torch.Tensor:
        # compute number of steps to reach n_timeframes (also the number of chunks)
        n_chunks = self.total_timeframes_per_case - max(self.init_sensor_timeframe_indices)
        # prepare fullstate timeframe indices (stochastic)
        torch.random.manual_seed(seed=seed)
        # TODO: each chunk could multiply to multiple samples
        random_init_timeframe_indices: torch.Tensor = torch.randperm(
            n=max(self.init_sensor_timeframe_indices)
        )[:self.n_fullstate_timeframes_per_chunk]
        fullstate_timeframe_indices: torch.Tensor = random_init_timeframe_indices + torch.arange(n_chunks).unsqueeze(1)
        assert fullstate_timeframe_indices.shape == (n_chunks, self.n_fullstate_timeframes_per_chunk)
        return fullstate_timeframe_indices


if __name__ == '__main__':
    # sensor_generator = LHS(spatial_shape=(140, 240), n_sensors=32)
    sensor_generator = AroundCylinder(resolution=(140, 240), n_sensors=32)
    # embedding_generator = Mask()
    embedding_generator = Voronoi(weighted=False)

    self = CFDDataset(
        root='./bc', 
        init_sensor_timeframe_indices=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        n_fullstate_timeframes_per_chunk=10,
        n_samplings_per_chunk=5,
        resolution=(140, 240),
        sensor_generator=sensor_generator, 
        embedding_generator=embedding_generator,
        seed=1,
    )
    sensor_timeframe_tensor, sensor_tensor, fullstate_timeframe_tensor, fullstate_tensor = self[500]
    print(sensor_timeframe_tensor.shape)
    print(sensor_tensor.shape)
    print(fullstate_timeframe_tensor.shape)
    print(fullstate_tensor.shape)


