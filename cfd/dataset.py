import os
import sys
from typing import List, Tuple, Dict, Any, Literal
import shutil
from tqdm import tqdm
import json

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from cfd.sensors import LHS, AroundCylinder
from cfd.embedding import Voronoi, Mask, Vector


class DatasetMixin:

    def load2tensor(self, case_dir: str) -> torch.Tensor:
        return torch.stack(
            tensors=[
                torch.from_numpy(np.load(os.path.join(case_dir, 'u.npy'))).cuda(),
                torch.from_numpy(np.load(os.path.join(case_dir, 'v.npy'))).cuda(),
            ],
            dim=1
        ).float()

    def prepare_sensor_timeframes(self) -> torch.IntTensor:
        # prepare sensor timeframes (fixed)
        sensor_timeframes: torch.Tensor = (
            torch.tensor(self.init_sensor_timeframes, device='cuda') + torch.arange(self.n_chunks, device='cuda').unsqueeze(1)
        )
        assert sensor_timeframes.shape == (self.n_chunks, len(self.init_sensor_timeframes))
        return sensor_timeframes.int()

    def prepare_fullstate_timeframes(self, seed: int | None = None, init_fullstate_timeframes: int | None = None) -> torch.IntTensor:
        assert seed is not None or init_fullstate_timeframes is not None, 'must be either deterministic or random'
        if seed is None and init_fullstate_timeframes is not None:    # deterministic
            fullstate_timeframes: torch.Tensor = (
                torch.arange(self.n_chunks, device='cuda').unsqueeze(1) 
                + torch.tensor(init_fullstate_timeframes, device='cuda').unsqueeze(0)
            )
            assert fullstate_timeframes.shape == (self.n_chunks, self.n_fullstate_timeframes_per_chunk)
            return fullstate_timeframes
        
        else:
            assert seed is not None, 'seed must be specified when target frames are generated randomly'
            fullstate_timeframes: torch.Tensor = torch.empty((self.n_chunks, self.n_fullstate_timeframes_per_chunk), dtype=torch.int, device='cuda')
            for chunk_idx in range(self.n_chunks):
                torch.random.manual_seed(seed + chunk_idx)
                random_init_timeframes: torch.Tensor = torch.randperm(
                    n=max(self.init_sensor_timeframes), device='cuda'
                )[:self.n_fullstate_timeframes_per_chunk].sort()[0]
                fullstate_timeframes[chunk_idx] = random_init_timeframes + chunk_idx

            assert fullstate_timeframes.shape == (self.n_chunks, self.n_fullstate_timeframes_per_chunk)
            return fullstate_timeframes


class CFDDataset(Dataset, DatasetMixin):

    def __init__(
        self, 
        root: str, 
        init_sensor_timeframes: List[int],
        n_fullstate_timeframes_per_chunk: int,
        n_samplings_per_chunk: int,
        resolution: Tuple[int, int],
        n_sensors: int,
        dropout_probabilities: List[float],
        noise_level: float,
        sensor_generator: Literal['LHS', 'AroundCylinder'], 
        embedding_generator: Literal['Voronoi', 'Mask', 'Vector'],
        init_fullstate_timeframes: List[int] | None,
        seed: int,
    ) -> None:
        
        super().__init__()
        self.case_directories: List[str] = sorted([os.path.join(root, case_dir) for case_dir in os.listdir(root)])
        self.root: str = root
        self.init_sensor_timeframes: List[int] = init_sensor_timeframes
        self.n_fullstate_timeframes_per_chunk: int = n_fullstate_timeframes_per_chunk
        self.n_samplings_per_chunk: int = n_samplings_per_chunk
        self.resolution: Tuple[int, int] = resolution
        self.n_sensors: int = n_sensors
        self.dropout_probabilities: List[float] = dropout_probabilities
        self.noise_level: float = noise_level
        self.init_fullstate_timeframes: List[int] | None = init_fullstate_timeframes
        self.seed: int = seed
        self.is_random_fullstate_frames: bool = init_fullstate_timeframes is None

        self.H, self.W = resolution
        self.n_sensor_timeframes_per_chunk: int = len(init_sensor_timeframes)
        self.total_timeframes_per_case: int = np.load(os.path.join(self.case_directories[-1], 'u.npy')).shape[0]

        self.dest: str = os.path.join('tensors', os.path.basename(root))
        self.sensor_timeframes_dest: str = os.path.join(self.dest, 'sensor_timeframes')
        self.sensor_values_dest: str = os.path.join(self.dest, 'sensor_values')
        self.fullstate_timeframes_dest: str = os.path.join(self.dest, 'fullstate_timeframes')
        self.fullstate_values_dest: str = os.path.join(self.dest, 'fullstate_values')
        self.sensor_positions_dest: str = os.path.join(self.dest, 'sensor_positions')
        self.metadata_dest: str = os.path.join(self.dest, 'metadata')

        if not self.is_random_fullstate_frames:
            # NOTE: fullstate frames are deterministically generated
            if n_fullstate_timeframes_per_chunk != len(init_fullstate_timeframes):
                raise ValueError(
                    f'n_fullstate_timeframes_per_chunk should be logically set to len(init_fullstate_timeframes) when sensors are generated deterministically, '
                    f'get: n_fullstate_timeframes_per_chunk={n_fullstate_timeframes_per_chunk} and init_fullstate_timeframes={init_fullstate_timeframes}'
                )
            if n_samplings_per_chunk != 1:
                raise ValueError(
                    f'n_samplings_per_chunk should be logically set to 1 when sensors are generated deterministically, '
                    f'get: {n_samplings_per_chunk}'
                )

        # NOTE: self.n_chunks is the number of samples in one case
        self.n_chunks: int = self.total_timeframes_per_case - max(self.init_sensor_timeframes)

        if sensor_generator == 'LHS':
            self.sensor_generator = LHS(n_sensors=n_sensors)
            self.sensor_generator.seed = seed
            self.sensor_generator.resolution = resolution
            self.sensor_positions = self.sensor_generator()
        else:
            self.sensor_generator = AroundCylinder(n_sensors=n_sensors)
            self.sensor_generator.seed = seed
            self.sensor_generator.resolution = resolution
            self.sensor_positions = self.sensor_generator(
                hw_meters=(0.14, 0.24), center_hw_meters=(0.07, 0.065), radius_meters=0.03,
            )
        
        assert self.sensor_positions.shape == (self.sensor_generator.n_sensors, 2)
        if embedding_generator == 'Mask':
            self.embedding_generator = Mask(
                resolution=resolution, sensor_positions=self.sensor_positions, 
                dropout_probabilities=dropout_probabilities, noise_level=noise_level,
            )
        elif embedding_generator == 'Voronoi':
            self.embedding_generator = Voronoi(
                resolution=resolution, sensor_positions=self.sensor_positions, 
                dropout_probabilities=dropout_probabilities, noise_level=noise_level,
            )
        else:
            self.embedding_generator = Vector(
                resolution=resolution, sensor_positions=self.sensor_positions, 
                dropout_probabilities=dropout_probabilities, noise_level=noise_level,
            )

        self.sensor_timeframes = self.prepare_sensor_timeframes()
        self.case_names: List[str] = []
        self.sampling_ids: List[int] = []
        self.__write2disk()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prefix: str = f'_{self.case_names[idx]}_{self.sampling_ids[idx]}_'
        suffix: str = str(idx).zfill(6)
        sensor_timeframe_tensor: torch.Tensor = torch.load(
            os.path.join(self.sensor_timeframes_dest, f'st{prefix}{suffix}.pt'), 
            weights_only=True
        )
        sensor_tensor: torch.Tensor = torch.load(
            os.path.join(self.sensor_values_dest, f'sv{prefix}{suffix}.pt'), 
            weights_only=True
        ).float()
        fullstate_timeframe_tensor: torch.Tensor = torch.load(
            os.path.join(self.fullstate_timeframes_dest, f'ft{prefix}{suffix}.pt'), 
            weights_only=True
        )
        fullstate_tensor: torch.Tensor = torch.load(
            os.path.join(self.fullstate_values_dest, f'fv{prefix}{suffix}.pt'), 
            weights_only=True
        ).float()
        case_name: str = self.case_names[idx]
        sampling_id: int = self.sampling_ids[idx]
        return sensor_timeframe_tensor, sensor_tensor, fullstate_timeframe_tensor, fullstate_tensor, case_name, sampling_id
    
    def __len__(self) -> int:
        return len([f for f in os.listdir(self.fullstate_values_dest) if f.endswith('.pt')])

    def __write2disk(self) -> None:
        # prepare dest directories
        if os.path.isdir(self.dest): shutil.rmtree(self.dest)
        os.makedirs(name=self.sensor_timeframes_dest, exist_ok=True)
        os.makedirs(name=self.sensor_values_dest, exist_ok=True)
        os.makedirs(name=self.fullstate_timeframes_dest, exist_ok=True)
        os.makedirs(name=self.fullstate_values_dest, exist_ok=True)
        os.makedirs(name=self.sensor_positions_dest, exist_ok=True)
        os.makedirs(name=self.metadata_dest, exist_ok=True)
        
        # save position tensors for reference
        torch.save(obj=self.sensor_positions, f=os.path.join(self.sensor_positions_dest, 'pos.pt'))

        sensor_timeframes_list: List[List[int]] = []
        fullstate_timeframes_list: List[List[int]] = []
        for case_id, case_dir in enumerate(self.case_directories):
            data: torch.Tensor = torch.stack(
                tensors=[
                    torch.from_numpy(np.load(os.path.join(case_dir, 'u.npy'))).cuda(),
                    torch.from_numpy(np.load(os.path.join(case_dir, 'v.npy'))).cuda(),
                ],
                dim=1
            ).float()
            
            # sensor data
            sensor_frame_data: torch.Tensor = data[self.sensor_timeframes]
            # resize sensor frames (original resolution is 64 x 64, which is not proportional to 0.14m x 0.24m)
            sensor_frame_data = F.interpolate(input=sensor_frame_data.flatten(0, 1), size=self.resolution, mode='bicubic')
            sensor_frame_data = sensor_frame_data.reshape(self.n_chunks, self.n_sensor_timeframes_per_chunk, 2, self.H, self.W)
            for sampling_id in range(self.n_samplings_per_chunk):
                # fullstate data
                if self.is_random_fullstate_frames:
                    print('Randomly generating fullstate frames')
                    fullstate_timeframes: torch.Tensor = self.prepare_fullstate_timeframes(seed=self.seed + case_id + sampling_id)
                else:
                    print('Deterministically generating fullstate frames')
                    fullstate_timeframes: torch.Tensor = self.prepare_fullstate_timeframes(
                        init_fullstate_timeframes=self.init_fullstate_timeframes
                    )

                fullstate_data: torch.Tensor = data[fullstate_timeframes]
                # resize fullstate frames
                fullstate_data = F.interpolate(input=fullstate_data.flatten(0, 1), size=self.resolution, mode='bicubic')
                fullstate_data = fullstate_data.reshape(self.n_chunks, self.n_fullstate_timeframes_per_chunk, 2, self.H, self.W)
                assert fullstate_data.shape == (self.n_chunks, self.n_fullstate_timeframes_per_chunk, 2, self.H, self.W)
                assert fullstate_timeframes.shape == (self.n_chunks, self.n_fullstate_timeframes_per_chunk)
                # compute sensor data for entire space
                sensor_data: torch.Tensor = self.embedding_generator(data=sensor_frame_data, seed=self.seed + case_id + sampling_id)
                if isinstance(self.embedding_generator, (Voronoi, Mask)):
                    assert sensor_data.shape == (self.n_chunks, self.n_sensor_timeframes_per_chunk, 2, self.H, self.W)
                else:
                    assert sensor_data.shape == (self.n_chunks, self.n_sensor_timeframes_per_chunk, 2, self.n_sensors)
                # Write each sample to disk
                for idx in tqdm(range(self.n_chunks), desc=f'Case {case_id + 1} | Sampling {sampling_id + 1}: '):
                    # case name & sampling id
                    case_name: str = os.path.basename(case_dir)
                    self.case_names.append(case_name)
                    self.sampling_ids.append(sampling_id)
                    prefix: str = f'_{case_name}_{sampling_id}_'
                    # indexes
                    true_idx: int = idx + sampling_id * self.n_chunks + case_id * self.n_chunks * self.n_samplings_per_chunk
                    suffix = str(true_idx).zfill(6)
                    # save sensor timeframes, sensor value (dynamic to chunks, but constant to samplings)
                    sensor_timeframes_list.append(self.sensor_timeframes[idx].tolist())
                    torch.save(obj=self.sensor_timeframes[idx].clone(), f=os.path.join(self.sensor_timeframes_dest, f'st{prefix}{suffix}.pt'))
                    torch.save(obj=sensor_data[idx].clone(), f=os.path.join(self.sensor_values_dest, f'sv{prefix}{suffix}.pt'))
                    # save sensor value, fullstate timeframes, fullstate data (fully dynamic)
                    fullstate_timeframes_list.append(fullstate_timeframes[idx].tolist())
                    torch.save(obj=fullstate_timeframes[idx].clone(), f=os.path.join(self.fullstate_timeframes_dest, f'ft{prefix}{suffix}.pt'))
                    torch.save(obj=fullstate_data[idx].clone(), f=os.path.join(self.fullstate_values_dest, f'fv{prefix}{suffix}.pt'))
                
                # manual garbage collection to optimize GPU RAM, otherwise likely lead to OutOfMemoryError
                del fullstate_data, sensor_data
        
        assert len(self.case_names) == len(self.sampling_ids) == len(sensor_timeframes_list) == len(fullstate_timeframes_list)
        records: List[Dict[str, Any]] = [
            {
                'case_name': case_name, 'sampling_id': sampling_id, 
                'sensor_timeframes': sensor_timeframes, 'fullstate_timeframes': fullstate_timeframes,
            }
            for case_name, sampling_id, sensor_timeframes, fullstate_timeframes in zip(
                self.case_names, self.sampling_ids, sensor_timeframes_list, fullstate_timeframes_list
            )
        ]
        with open(os.path.join(self.metadata_dest, 'metadata.json'), 'w') as f:
            json.dump(obj=records, fp=f, indent=2)


