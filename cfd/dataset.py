import os
import sys
from typing import List, Tuple, Dict, Any
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
                torch.from_numpy(np.load(os.path.join(case_dir, 'u.npy'))),
                torch.from_numpy(np.load(os.path.join(case_dir, 'v.npy')))
            ],
            dim=1
        ).float()

    def prepare_sensor_timeframes(self) -> torch.IntTensor:
        # compute number of steps to reach n_timeframes (also the number of chunks)
        n_chunks = self.total_timeframes_per_case - max(self.init_sensor_timeframes)
        # prepare sensor timeframes (fixed)
        sensor_timeframes: torch.Tensor = torch.tensor(self.init_sensor_timeframes) + torch.arange(n_chunks).unsqueeze(1)
        assert sensor_timeframes.shape == (n_chunks, len(self.init_sensor_timeframes))
        return sensor_timeframes.int()

    def prepare_fullstate_timeframes(self, seed: int) -> torch.IntTensor:
        n_chunks = self.total_timeframes_per_case - max(self.init_sensor_timeframes)
        fullstate_timeframes: torch.Tensor = torch.empty((n_chunks, self.n_fullstate_timeframes_per_chunk), dtype=torch.int)
        for chunk_idx in range(n_chunks):
            torch.random.manual_seed(seed + chunk_idx)
            random_init_timeframes = torch.randperm(
                n=max(self.init_sensor_timeframes)
            )[:self.n_fullstate_timeframes_per_chunk].sort()[0]
            fullstate_timeframes[chunk_idx] = random_init_timeframes + chunk_idx

        assert fullstate_timeframes.shape == (n_chunks, self.n_fullstate_timeframes_per_chunk)
        return fullstate_timeframes


class CFDDataset(Dataset, DatasetMixin):

    def __init__(
        self, 
        root: str, 
        init_sensor_timeframes: List[int],
        n_fullstate_timeframes_per_chunk: int,
        n_samplings_per_chunk: int,
        resolution: Tuple[int, int],
        sensor_generator: LHS | AroundCylinder, 
        embedding_generator: Voronoi | Mask,
        seed: int,
        already_preloaded: bool
    ) -> None:
        
        super().__init__()
        self.case_directories: List[str] = sorted(
            [os.path.join(root, case_dir) for case_dir in os.listdir(root)]
        )

        self.root: str = root
        self.init_sensor_timeframes: List[int] = init_sensor_timeframes
        self.n_fullstate_timeframes_per_chunk: int = n_fullstate_timeframes_per_chunk
        self.n_samplings_per_chunk: int = n_samplings_per_chunk
        self.resolution: Tuple[int, int] = resolution
        self.sensor_generator: LHS | AroundCylinder = sensor_generator
        self.sensor_generator.seed = seed
        self.embedding_generator: Voronoi | Mask = embedding_generator
        self.seed: int = seed
        self.already_preloaded: bool = already_preloaded

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
        
        self.sensor_timeframes = self.prepare_sensor_timeframes()
        if self.already_preloaded:
            self.sensor_positions: torch.Tensor = torch.load(
                f=os.path.join(self.sensor_positions_dest, 'pos.pt'), weights_only=True
            )
            with open(os.path.join(self.metadata_dest, 'metadata.json'), 'r') as f:
                records: List[Dict[str, Any]] = json.load(f)
                self.case_names: List[str] = [record['case_name'] for record in records]
                self.sampling_ids: List[int] = [record['sampling_id'] for record in records]
        else:
            self.sensor_positions: torch.Tensor
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

        # prepare sensor positions
        self.sensor_generator.resolution = self.resolution
        self.sensor_generator.seed = self.seed
        if isinstance(self.sensor_generator, LHS):
            self.sensor_positions = self.sensor_generator()
        else:
            self.sensor_positions = self.sensor_generator(
                hw_meters=(0.14, 0.24), center_hw_meters=(0.07, 0.065), radius_meters=0.03,
            )

        assert self.sensor_positions.shape == (self.sensor_generator.n_sensors, 2)
        # save self.sensor_positions for self.already_preloaded = True
        torch.save(obj=self.sensor_positions, f=os.path.join(self.sensor_positions_dest, 'pos.pt'))

        sensor_timeframes_list: List[List[int]] = []
        fullstate_timeframes_list: List[List[int]] = []
        for case_id, case_dir in enumerate(self.case_directories):
            data: torch.Tensor = torch.stack(
                tensors=[
                    torch.from_numpy(np.load(os.path.join(case_dir, 'u.npy'))),
                    torch.from_numpy(np.load(os.path.join(case_dir, 'v.npy')))
                ],
                dim=1
            ).float()
            
            # sensor data
            sensor_data: torch.Tensor = data[self.sensor_timeframes]
            n_chunks: int = sensor_data.shape[0]
            # resize sensor frames (original resolution is 64 x 64, which is not proportional to 0.14m x 0.24m)
            sensor_data = F.interpolate(input=sensor_data.flatten(0, 1), size=self.resolution, mode='bicubic')
            sensor_data = sensor_data.reshape(n_chunks, self.n_sensor_timeframes_per_chunk, 2, self.H, self.W)
            # compute sensor data for entire space
            sensor_data = self.embedding_generator(data=sensor_data, sensor_positions=self.sensor_positions)
            assert sensor_data.shape == (n_chunks, self.n_sensor_timeframes_per_chunk, 2, self.H, self.W)

            for sampling_id in range(self.n_samplings_per_chunk):
                # fullstate data
                fullstate_timeframes: torch.Tensor = self.prepare_fullstate_timeframes(seed=self.seed + case_id + sampling_id)
                fullstate_data: torch.Tensor = data[fullstate_timeframes]
                # resize fullstate frames
                fullstate_data = F.interpolate(input=fullstate_data.flatten(0, 1), size=self.resolution, mode='bicubic')
                fullstate_data = fullstate_data.reshape(n_chunks, self.n_fullstate_timeframes_per_chunk, 2, self.H, self.W)

                assert fullstate_data.shape == (n_chunks, self.n_fullstate_timeframes_per_chunk, 2, self.H, self.W)
                assert fullstate_timeframes.shape == (n_chunks, self.n_fullstate_timeframes_per_chunk)

                for idx in tqdm(range(n_chunks), desc=f'Case {case_id + 1} | Sampling {sampling_id + 1}: '):
                    # case name & sampling id
                    case_name: str = os.path.basename(case_dir)
                    self.case_names.append(case_name)
                    self.sampling_ids.append(sampling_id)
                    prefix: str = f'_{case_name}_{sampling_id}_'
                    # indexes
                    true_idx: int = idx + sampling_id * n_chunks + case_id * n_chunks * self.n_samplings_per_chunk
                    suffix = str(true_idx).zfill(6)
                    # save sensor timeframes, sensor value (dynamic to chunks, but constant to samplings)
                    sensor_timeframes_list.append(self.sensor_timeframes[idx].tolist())
                    torch.save(obj=self.sensor_timeframes[idx].clone(), f=os.path.join(self.sensor_timeframes_dest, f'st{prefix}{suffix}.pt'))
                    torch.save(obj=sensor_data[idx].clone(), f=os.path.join(self.sensor_values_dest, f'sv{prefix}{suffix}.pt'))
                    # save sensor value, fullstate timeframes, fullstate data (fully dynamic)
                    fullstate_timeframes_list.append(fullstate_timeframes[idx].tolist())
                    torch.save(obj=fullstate_timeframes[idx].clone(), f=os.path.join(self.fullstate_timeframes_dest, f'ft{prefix}{suffix}.pt'))
                    torch.save(obj=fullstate_data[idx].clone(), f=os.path.join(self.fullstate_values_dest, f'fv{prefix}{suffix}.pt'))
        
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


