import os
from typing import List

import numpy as np
import torch

from torch.utils.data import Dataset


class CFDDataset(Dataset):

    def __init__(self, root: str):
        super().__init__()
        self.root: str = root
        self.case_directories: List[str] = sorted([os.path.join(root, casedir) for casedir in os.listdir(root)])
        self.data: torch.Tensor = self.__load2ram()

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]
    
    def __len__(self) -> int:
        return self.data.shape[0]

    def __load2ram(self):
        case_tensors: List[torch.Tensor] = [
            torch.stack(
                tensors=[
                    torch.from_numpy(np.load(os.path.join(casedir, 'u.npy'))),
                    torch.from_numpy(np.load(os.path.join(casedir, 'v.npy')))
                ],
                dim=1
            )
            for casedir in self.case_directories
        ]
        return torch.cat(tensors=case_tensors, dim=0) # (N, 2, H, W)


if __name__ == '__main__':
    self = CFDDataset('./bc')





