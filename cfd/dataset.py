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
from cfd.embedding import Voronoi, Mask


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

    def prepare_fullstate_timeframes(self, seed: int | None = None, init_fullstate_timeframe: int = -1) -> torch.IntTensor:
        assert seed is not None or init_fullstate_timeframe != -1, 'must be either deterministic or random'
        if seed is None and init_fullstate_timeframe != -1:    # deterministic
            assert self.n_fullstate_timeframes_per_chunk == 1, (
                f'n_fullstate_timeframes_per_chunk should be logically set to 1 when target frames are generated deterministically '
                f'(otherwise it contains overlapping frames), '
                f'get: {self.n_fullstate_timeframes_per_chunk}'
            )
            fullstate_timeframes: torch.Tensor = torch.arange(self.n_chunks, device='cuda').unsqueeze(1) + init_fullstate_timeframe
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
        sensor_generator: Literal['LHS', 'AroundCylinder'], 
        embedding_generator: Literal['Voronoi', 'Mask'],
        init_fullstate_timeframe: int | None,
        seed: int,
        already_preloaded: bool
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
        self.init_fullstate_timeframe: int | None = init_fullstate_timeframe
        self.seed: int = seed
        self.already_preloaded: bool = already_preloaded
        self.is_random_fullstate_frames: bool = init_fullstate_timeframe is None

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

        if not self.already_preloaded:
            if init_fullstate_timeframe is not None:
                # NOTE: deterministically generate fullstate frames
                if n_fullstate_timeframes_per_chunk != 1:
                    raise ValueError(
                        f'n_fullstate_timeframes_per_chunk should be logically set to 1 when sensors are generated deterministically, '
                        f'(otherwise it contains overlapping frames), '
                        f'get: {n_fullstate_timeframes_per_chunk}'
                    )
                if n_samplings_per_chunk != 1:
                    raise ValueError(
                        f'n_samplings_per_chunk should be logically set to 1 when sensors are generated deterministically, '
                        f'get: {n_samplings_per_chunk}'
                    )
                self.n_chunks: int = self.total_timeframes_per_case - max(self.init_sensor_timeframes + [init_fullstate_timeframe])  # usable for time extrapolation
            else:
                self.n_chunks: int = self.total_timeframes_per_case - max(self.init_sensor_timeframes)
            # NOTE: self.n_chunks is the number of samples in one case

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
                    resolution=resolution, sensor_positions=self.sensor_positions, dropout_probabilities=dropout_probabilities
                )
            else:
                self.embedding_generator = Voronoi(
                    resolution=resolution, sensor_positions=self.sensor_positions, dropout_probabilities=dropout_probabilities
                )

            self.sensor_timeframes = self.prepare_sensor_timeframes()
            self.case_names: List[str] = []
            self.sampling_ids: List[int] = []
            self.__write2disk()

        else:
            self.sensor_positions: torch.Tensor = torch.load(
                f=os.path.join(self.sensor_positions_dest, 'pos.pt'), weights_only=True
            )
            with open(os.path.join(self.metadata_dest, 'metadata.json'), 'r') as f:
                records: List[Dict[str, Any]] = json.load(f)
                self.case_names: List[str] = [record['case_name'] for record in records]
                self.sampling_ids: List[int] = [record['sampling_id'] for record in records]

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
        if os.path.exists(self.dest): 
            confirmed: bool = input(
                f'Enter "yes" to permanently DELETE the loaded dataset in {self.dest} and reload over again: '
            ) == 'yes'
            if confirmed:
                shutil.rmtree(self.dest)
            else:
                print('Terminated')
                sys.exit(1)

        os.makedirs(name=self.sensor_timeframes_dest, exist_ok=True)
        os.makedirs(name=self.sensor_values_dest, exist_ok=True)
        os.makedirs(name=self.fullstate_timeframes_dest, exist_ok=True)
        os.makedirs(name=self.fullstate_values_dest, exist_ok=True)
        os.makedirs(name=self.sensor_positions_dest, exist_ok=True)
        os.makedirs(name=self.metadata_dest, exist_ok=True)

        # save self.sensor_positions for self.already_preloaded = True
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
                        init_fullstate_timeframe=self.init_fullstate_timeframe
                    )

                fullstate_data: torch.Tensor = data[fullstate_timeframes]
                # resize fullstate frames
                fullstate_data = F.interpolate(input=fullstate_data.flatten(0, 1), size=self.resolution, mode='bicubic')
                fullstate_data = fullstate_data.reshape(self.n_chunks, self.n_fullstate_timeframes_per_chunk, 2, self.H, self.W)
                assert fullstate_data.shape == (self.n_chunks, self.n_fullstate_timeframes_per_chunk, 2, self.H, self.W)
                assert fullstate_timeframes.shape == (self.n_chunks, self.n_fullstate_timeframes_per_chunk)
                # compute sensor data for entire space
                sensor_data: torch.Tensor = self.embedding_generator(data=sensor_frame_data, seed=self.seed + case_id + sampling_id)
                assert sensor_data.shape == (self.n_chunks, self.n_sensor_timeframes_per_chunk, 2, self.H, self.W)
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
        
        # save self.case_names for self.already_preloaded = True
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



if __name__ == '__main__':
    # NOTE: How to load precomputed dataset from disk
    self = CFDDataset(
        root='data/train',
        # root='data/test',
        init_sensor_timeframes=[0, 5, 10, 15, 20],
        n_fullstate_timeframes_per_chunk=1,
        n_samplings_per_chunk=1,
        resolution=(140, 240),
        n_sensors=32,
        dropout_probabilities=[0.05, 0.04, 0.03, 0.02, 0.01],
        sensor_generator='LHS',
        embedding_generator='Voronoi',
        init_fullstate_timeframe=None,
        seed=1,
        already_preloaded=False,    # set to False for first time load
    )
    # first sample
    sensor_timeframe_tensor, sensor_tensor, fullstate_timeframe_tensor, fullstate_tensor, case_name, sampling_id = self[0]
