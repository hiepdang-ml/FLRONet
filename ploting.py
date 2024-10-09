import os
from typing import List, Tuple, Optional, Callable

import datetime as dt
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


# TODO: plot sensor positions
def plot_groundtruth_2d(
    groundtruth: torch.Tensor,
    reduction: Callable[[torch.Tensor], torch.Tensor] | None = None,
    resolution: Tuple[int, int] | None = None,
) -> None:
    
    assert groundtruth.ndim == 3   # (n_fields, x_resolution, y_resolution)

    if reduction is not None:
        groundtruth: torch.Tensor = reduction(groundtruth)

    assert groundtruth.shape[0] == 1, (
        f'All physical fields must be aggregated to a single field for visualization, '
        f'got groundtruth.shape[0]={groundtruth.shape[0]}'
    )
    # Prepare output directory and move tensor to CPU
    destination_directory: str = './plots/groundtruths'
    os.makedirs(destination_directory, exist_ok=True)
    groundtruth = groundtruth.to(device=torch.device('cpu'))

    # Resize:
    if resolution is not None:
        groundtruth: torch.Tensor = F.interpolate(input=groundtruth.unsqueeze(dim=0), size=resolution, mode='bicubic').squeeze(dim=0)

    # Ensure that the plot respect the tensor's shape
    x_res: int = groundtruth.shape[1]
    y_res: int = groundtruth.shape[2]
    aspect_ratio: float = x_res / y_res

    # Set plot configuration
    cmap: str = 'jet'

    for idx in range(groundtruth.shape[0]):
        field: torch.Tensor = groundtruth[idx]
        figwidth: float = 8.
        fig, ax = plt.subplots(figsize=(figwidth, figwidth * aspect_ratio))
        ax.imshow(
            field.squeeze(dim=0),
            origin="lower",
            vmin=0, vmax=field.max().item(),
            cmap=cmap,
        )
        ax.set_title(f'$groundtruth$', fontsize=15)
        
        fig.tight_layout()
        timestamp: dt.datetime = dt.datetime.now()
        fig.savefig(
            f"{destination_directory}/{timestamp.strftime('%Y%m%d%H%M%S')}"
            f"{timestamp.microsecond // 1000:03d}.png"
        )
        plt.close(fig)    



if __name__ == '__main__':
    from functional import compute_velocity_field
    from datasets import CFDDataset
    self = CFDDataset('./bc')
    gt = self[800]
    plot_groundtruth_2d(
        groundtruth=gt,
        reduction=lambda x: compute_velocity_field(x, dim=0),
        resolution=(128, 256),
    )
    
