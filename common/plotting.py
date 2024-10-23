import os
from typing import Callable

import datetime as dt
import matplotlib
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
    title: str = '',
    filename: str = '',
) -> None:
    
    if sensor_positions is not None:
        assert sensor_positions.ndim == 2    # (n_sensors, 2)
        n_sensors: int = sensor_positions.shape[0]
        sensor_positions = sensor_positions.cpu()

    # Collect frames that are not None
    frames_to_plot = []
    chart_titles = []
    if sensor_frame is not None:
        frames_to_plot.append(sensor_frame)
        chart_titles.append(f"Sensor Value")
        
    if fullstate_frame is not None:
        frames_to_plot.append(fullstate_frame)
        chart_titles.append(f"Full State")
        
    if reconstruction_frame is not None:
        frames_to_plot.append(reconstruction_frame)
        chart_titles.append(f"Reconstruction")

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
    if fullstate_frame is not None:
        max_value: float = fullstate_frame.max().item()
    else:
        max_value: float = max([frame.max().item() for frame in frames_to_plot])

    # matplotlib.colors.Normalize(vmin=0, vmax=max_value)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=2.5)
    for frame, ax, chart_title in zip(frames_to_plot, axs, chart_titles):
        im = ax.imshow(
            frame.squeeze(dim=0),
            origin="lower",
            norm=norm,
            cmap='jet',
        )
        cbar = ax.figure.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=10)

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
        ax.set_title(f'{chart_title}', fontsize=14)

    fig.suptitle(title, fontsize=15)
    # Finalize and save the figure
    fig.tight_layout()
    destination_directory = './plots'
    os.makedirs(destination_directory, exist_ok=True)
    timestamp: dt.datetime = dt.datetime.now()
    if not filename:
        filename =f"{destination_directory}/{timestamp.strftime('%Y%m%d%H%M%S')}{timestamp.microsecond // 1000:03d}.png"
    else:
        fig.savefig(f"{destination_directory}/{filename}.png")
    plt.close(fig)


