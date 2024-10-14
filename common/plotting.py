import os
from typing import Callable

import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torch.nn.functional as F


def plot_frame(
    sensor_positions: torch.Tensor | None = None,
    sensor_frame: torch.Tensor | None = None,
    fullstate_frame: torch.Tensor | None = None,
    reconstruction_frame: torch.Tensor | None = None,
    reduction: Callable[[torch.Tensor], torch.Tensor] | None = None,
    prefix: str = '',
    suffix: str = '',
) -> None:
    
    if sensor_positions is not None:
        assert sensor_positions.ndim == 2    # (n_sensors, 2)
        n_sensors: int = sensor_positions.shape[0]

    # Collect frames that are not None
    frames_to_plot = []
    titles = []

    if sensor_frame is not None:
        frames_to_plot.append(sensor_frame)
        titles.append(f"{prefix} Sensor Frame {suffix}")
        
    if fullstate_frame is not None:
        frames_to_plot.append(fullstate_frame)
        titles.append(f"{prefix} Full State Frame {suffix}")
        
    if reconstruction_frame is not None:
        frames_to_plot.append(reconstruction_frame)
        titles.append(f"{prefix} Reconstruction Frame {suffix}")

    frame_shapes = [frame.shape for frame in frames_to_plot]
    assert all(shape == frame_shapes[0] for shape in frame_shapes), "All provided frames must have the same shape."

    # Process the reduction and resolution
    if reduction is not None:
        frames_to_plot = [reduction(frame) for frame in frames_to_plot]

    assert all(frame.shape[0] == 1 for frame in frames_to_plot), (
        'All physical fields must be aggregated to a single field for visualization.'
    )

    # Move to cpu for ploting
    frames_to_plot = [frame.cpu() for frame in frames_to_plot]

    # Set up the plot
    num_plots = len(frames_to_plot)
    aspect_ratio: float = frames_to_plot[0].shape[1] / frames_to_plot[0].shape[2]
    figwidth: float = 8.0
    fig, axs = plt.subplots(num_plots, 1, figsize=(figwidth, figwidth * aspect_ratio * num_plots))

    if num_plots == 1:
        axs = [axs]  # Ensure axs is iterable if only one subplot
    
    # Plot each frame
    max_value: float = max([frame.max().item() for frame in frames_to_plot])
    for frame, ax, title in zip(frames_to_plot, axs, titles):
        ax.imshow(
            frame.squeeze(dim=0),
            origin="lower",
            vmin=0, vmax=max_value,
            cmap='jet',
        )
        if sensor_positions is not None:
            for sensor_x, sensor_y in sensor_positions:
                ax.add_patch(
                    patches.Rectangle(
                        xy=(sensor_y, sensor_x),
                        width=1, height=1,  # Size of the marker (1 pixel)
                        edgecolor='white',
                        facecolor='white',
                        fill=True
                    )
                )

        ax.set_title(f'{title}', fontsize=15)

    # Finalize and save the figure
    fig.tight_layout()
    destination_directory = './plots'
    os.makedirs(destination_directory, exist_ok=True)
    timestamp: dt.datetime = dt.datetime.now()
    fig.savefig(
        f"{destination_directory}/{timestamp.strftime('%Y%m%d%H%M%S')}"
        f"{timestamp.microsecond // 1000:03d}.png"
    )
    plt.close(fig)



if __name__ == '__main__':

    from common.functional import compute_velocity_field
    from cfd.dataset import CFDDataset
    from cfd.sensors import LHS, AroundCylinder
    from cfd.embedding import Mask, Voronoi
    
    # sensor_generator = LHS(spatial_shape=(140, 240), n_sensors=32)
    sensor_generator = AroundCylinder(resolution=(140, 240), n_sensors=64)
    # embedding_generator = Mask()
    embedding_generator = Voronoi(weighted=False)

    dataset = CFDDataset(
        root='./data/val', 
        init_sensor_timeframe_indices=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        n_fullstate_timeframes_per_chunk=10,
        n_samplings_per_chunk=1,
        resolution=(140, 240),
        sensor_generator=sensor_generator, 
        embedding_generator=embedding_generator,
        seed=1,
    )

    sensor_timeframe_tensor, sensor_tensor, fullstate_timeframe_tensor, fullstate_tensor = dataset[0]
    sensor_positions = dataset.sensor_positions

    plot_frame(
        sensor_positions=sensor_positions,
        sensor_frame=sensor_tensor[0],          # only get first frame
        fullstate_frame=fullstate_tensor[0],    # only get first frame
        reduction=lambda x: compute_velocity_field(x, dim=0),
    )
    
