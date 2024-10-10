import os
from typing import List, Tuple, Optional

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
        
        self.n_sensor_frames_per_chunk: int = len(init_sensor_frame_indices)
        self.n_fullstate_frames_per_chunk: int = len(init_fullstate_frame_indices)
        self.sensor_data, self.fullstate_data, self.rel_sensor_positions, self.abs_sensor_positions = self.__load2ram()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sensor_data[idx], self.fullstate_data[idx]
    
    def __len__(self) -> int:
        return self.sensor_data.shape[0]

    def __load2ram(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        sensor_data: List[torch.Tensor] = []
        fullstate_data: List[torch.Tensor] = []
        # for casedir in self.case_directories:
        for casedir in self.case_directories[:2]:
            data: torch.Tensor = torch.stack(
                tensors=[
                    torch.from_numpy(np.load(os.path.join(casedir, 'u.npy'))),
                    torch.from_numpy(np.load(os.path.join(casedir, 'v.npy')))
                ],
                dim=1
            )
            # sensor data
            sensor_tensor: torch.Tensor = data[self.sensor_frame_indices]
            n_chunks: int = sensor_tensor.shape[0]
            H, W = sensor_tensor.shape[-2:]
            assert sensor_tensor.shape == (n_chunks, self.n_sensor_frames_per_chunk, 2, H, W)
            # fullstate data
            fullstate_tensor: torch.Tensor = data[self.fullstate_frame_indices]
            assert fullstate_tensor.shape == (n_chunks, self.n_fullstate_frames_per_chunk, 2, H, W)
            # append
            sensor_data.append(sensor_tensor)
            fullstate_data.append(fullstate_tensor)

        sensor_data: torch.Tensor = torch.cat(tensors=sensor_data, dim=0)
        fullstate_data: torch.Tensor = torch.cat(tensors=fullstate_data, dim=0)
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

        # prepare sensor positions
        if isinstance(self.sensor_generator, LHS):
            rel_sensor_positions, abs_sensor_positions = self.sensor_generator.generate()
        else:
            rel_sensor_positions, abs_sensor_positions = self.sensor_generator.generate(
                hw_meters=(0.14, 0.24), center_hw_meters=(0.08, 0.08), radius_meters=0.01
            )

        # prepare embeddings for sensor data
        sensor_data = self.embedding_generator(data=sensor_data, sensor_positions=abs_sensor_positions)

        assert sensor_data.shape == (n_samples, self.n_sensor_frames_per_chunk, 2, H, W)
        assert fullstate_data.shape == (n_samples, self.n_fullstate_frames_per_chunk, 2, H, W)
        return sensor_data, fullstate_data, rel_sensor_positions, abs_sensor_positions
    
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
    )
    sensor_data, fullstate_data = self[500]
    print(sensor_data)
    print(fullstate_data)






