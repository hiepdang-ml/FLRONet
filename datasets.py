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

    # TODO: implement interpolation to arbitrary shape
    def __init__(
        self, 
        root: str, 
        init_fullstate_frame_indices: List[int],
        init_sensor_frame_indices: List[int],
        resolution: Tuple[int, int] | None,
        sensor_generator: LHS | AroundCylinder, 
        embedding_generator: Voronoi | Mask,
        already_preloaded: bool,
    ) -> None:
        
        super().__init__()
        self.case_directories: List[str] = sorted(
            [os.path.join(root, casedir) for casedir in os.listdir(root)]
        )

        self.root: str = root
        self.sensor_frame_indices, self.fullstate_frame_indices = self.__prepare_frame_indices(
            init_sensor_frame_indices, init_fullstate_frame_indices
        )
        self.resolution: Tuple[int, int] | None = resolution
        self.sensor_generator: LHS | AroundCylinder = sensor_generator
        self.embedding_generator: Voronoi | Mask = embedding_generator
        self.already_preloaded: bool = already_preloaded

        self.n_sensor_frames_per_chunk: int = len(init_sensor_frame_indices)
        self.n_fullstate_frames_per_chunk: int = len(init_fullstate_frame_indices)

        self.sensors_dest: str = os.path.join('tensors', 'sensors')
        self.fullstates_dest: str = os.path.join('tensors', 'fullstates')
        self.positions_dest: str = os.path.join('tensors', 'positions')
        
        if not already_preloaded:
            # expensive
            self.__write2disk()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        suffix: str = str(idx).zfill(6)
        sensor_tensor: torch.Tensor = torch.load(os.path.join(self.sensors_dest, f'S{suffix}.pt'))
        fullstate_tensor: torch.Tensor = torch.load(os.path.join(self.fullstates_dest, f'F{suffix}.pt'))
        return sensor_tensor, fullstate_tensor
    
    def __len__(self) -> int:
        return len([f for f in os.listdir(self.fullstates_dest) if f.endswith('.pt')])

    def __write2disk(self) -> None:

        # prepare dest directories
        shutil.rmtree('tensors')
        os.makedirs(name=self.sensors_dest, exist_ok=True)
        os.makedirs(name=self.fullstates_dest, exist_ok=True)
        os.makedirs(name=self.positions_dest, exist_ok=True)

        for caseid, casedir in enumerate(self.case_directories):
            data: torch.Tensor = torch.stack(
                tensors=[
                    torch.from_numpy(np.load(os.path.join(casedir, 'u.npy'))),
                    torch.from_numpy(np.load(os.path.join(casedir, 'v.npy')))
                ],
                dim=1
            )
            # sensor data
            sensor_data: torch.Tensor = data[self.sensor_frame_indices]
            n_chunks: int = sensor_data.shape[0]
            H, W = sensor_data.shape[-2:]
            assert sensor_data.shape == (n_chunks, self.n_sensor_frames_per_chunk, 2, H, W)
            # fullstate data
            fullstate_data: torch.Tensor = data[self.fullstate_frame_indices]
            assert fullstate_data.shape == (n_chunks, self.n_fullstate_frames_per_chunk, 2, H, W)

            n_samples: int = sensor_data.shape[0]
            assert sensor_data.shape == (n_samples, self.n_sensor_frames_per_chunk, 2, H, W)
            assert fullstate_data.shape == (n_samples, self.n_fullstate_frames_per_chunk, 2, H, W)

            # adjust resolution (original resolution is 64 x 64, which is not proportional to 0.14m x 0.24m)
            if self.resolution is not None:
                H, W = self.resolution
                # resize sensor
                sensor_data = F.interpolate(input=sensor_data.flatten(0, 1), size=self.resolution, mode='bicubic')
                sensor_data = sensor_data.reshape(n_samples, self.n_sensor_frames_per_chunk, 2, H, W)
                # resize fullstate
                fullstate_data = F.interpolate(input=fullstate_data.flatten(0, 1), size=self.resolution, mode='bicubic')
                fullstate_data = fullstate_data.reshape(n_samples, self.n_fullstate_frames_per_chunk, 2, H, W)
                fullstate_data = fullstate_data.float()

            # prepare sensor positions
            if isinstance(self.sensor_generator, LHS):
                sensor_positions = self.sensor_generator()
            else:
                sensor_positions = self.sensor_generator(
                    hw_meters=(0.14, 0.24), center_hw_meters=(0.08, 0.08), radius_meters=0.01
                )

            # only save position tensor once
            if caseid == 0:
                torch.save(obj=sensor_positions, f=os.path.join(self.positions_dest, f'pos.pt'))

            # prepare embeddings for sensor data
            sensor_data = self.embedding_generator(data=sensor_data, sensor_positions=sensor_positions)
            sensor_data = sensor_data.float()

            assert sensor_data.shape == (n_samples, self.n_sensor_frames_per_chunk, 2, H, W)
            assert fullstate_data.shape == (n_samples, self.n_fullstate_frames_per_chunk, 2, H, W)

            for idx in tqdm.tqdm(range(n_samples), desc=f'Case {caseid + 1}'):
                true_idx: int = idx + caseid * n_samples
                suffix: str = str(true_idx).zfill(6)
                torch.save(obj=sensor_data[idx].clone(), f=os.path.join(self.sensors_dest, f'S{suffix}.pt'))
                torch.save(obj=fullstate_data[idx].clone(), f=os.path.join(self.fullstates_dest, f'F{suffix}.pt'))
        
    def __prepare_frame_indices(
        self, 
        init_sensor_frame_indices: List[int], 
        init_fullstate_frame_indices: List[int]
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:

        # read any file to get number of frames
        n_frames: int = np.load(os.path.join(self.case_directories[0], 'u.npy')).shape[0]
        # compute number of steps to reach n_frames
        steps = n_frames - max(init_sensor_frame_indices)
        # prepare indices
        sensor_frame_indices: torch.Tensor = torch.tensor(init_sensor_frame_indices) + torch.arange(steps).unsqueeze(1)
        fullstate_frame_indices: torch.Tensor = torch.tensor(init_fullstate_frame_indices) + torch.arange(steps).unsqueeze(1)
        assert sensor_frame_indices.shape[0] == fullstate_frame_indices.shape[0]
        return sensor_frame_indices, fullstate_frame_indices


if __name__ == '__main__':
    # sensor_generator = LHS(spatial_shape=(64, 64), n_sensors=32)
    sensor_generator = AroundCylinder(spatial_shape=(64, 64), n_sensors=32)
    # embedding_generator = Mask()
    embedding_generator = Voronoi(weighted=False)

    self = CFDDataset(
        root='./bc', 
        init_fullstate_frame_indices=[0, 7, 13, 24, 36, 42, 50],
        init_sensor_frame_indices=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        resolution=(140, 240),
        sensor_generator=sensor_generator, 
        embedding_generator=embedding_generator,
        already_preloaded=True,
    )
    sensor_data, fullstate_data = self[500]
    print(sensor_data)
    print(fullstate_data)






