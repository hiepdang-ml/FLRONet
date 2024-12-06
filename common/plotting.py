import os
from typing import Callable

import datetime as dt
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import colormaps as cmaps

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
        sensor_frame = reduction(sensor_frame)
        frames_to_plot.append(sensor_frame)
        chart_titles.append(f"Sensor Value")
        
    if reconstruction_frame is not None:
        reconstruction_frame = reduction(reconstruction_frame)
        frames_to_plot.append(reconstruction_frame)
        chart_titles.append(f"Reconstruction")

    if fullstate_frame is not None:
        fullstate_frame = reduction(fullstate_frame)
        frames_to_plot.append(fullstate_frame)
        chart_titles.append(f"Full State")

    if reconstruction_frame is not None and fullstate_frame is not None:
        error_frame = reconstruction_frame - fullstate_frame
        frames_to_plot.append(error_frame)
        chart_titles.append(f"Error")

    frame_shapes = [frame.shape for frame in frames_to_plot]
    assert all(shape == frame_shapes[0] for shape in frame_shapes), "All provided frames must have the same shape."

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
    
    max_value: float = 10.
    for frame, ax, chart_title in zip(frames_to_plot, axs, chart_titles):
        if chart_title == 'Error':
            norm = matplotlib.colors.Normalize(vmin=-max_value, vmax=max_value)
        else:
            norm = matplotlib.colors.Normalize(vmin=0, vmax=max_value)

        im = ax.imshow(frame.squeeze(dim=0), origin="lower", norm=norm, cmap=cmaps.balance)
        # Display cmap bar
        # cbar = ax.figure.colorbar(im, ax=ax, orientation='vertical', fraction=0.027, pad=0.04)
        # cbar.ax.tick_params(labelsize=14)
        # if chart_title == 'Error':
        #     tick_values = list(range(int(cbar.vmin), int(cbar.vmax) + 1, 5))
        # else:
        #     tick_values = list(range(int(cbar.vmin), int(cbar.vmax) + 1, 2))
        # cbar.set_ticks(tick_values)

        if sensor_positions is not None:
            for sensor_x, sensor_y in sensor_positions:
                # mark sensors
                ax.add_patch(
                    patches.Circle(
                        xy=(sensor_y, sensor_x),
                        radius=1.3,  # Adjust the radius for dot size
                        edgecolor='black',
                        facecolor='white',
                        fill=True
                    )
                )
        # NOTE: setting for display in report
        ax.tick_params(labelbottom=False, labelleft=False)  # remove tick labels
        # remove ticks
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title, fontsize=15)
    # Ensure no margin
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    plt.subplots_adjust(left=0, right=1, top=0.99, bottom=0.01)
    plt.margins(0)
    # Save the figure
    destination_directory = './plots'
    os.makedirs(destination_directory, exist_ok=True)
    timestamp: dt.datetime = dt.datetime.now()
    if not filename:
        filename: str = f"{timestamp.strftime('%Y%m%d%H%M%S')}{timestamp.microsecond // 1000:03d}"

    # fig.savefig(os.path.join(destination_directory, f'{filename}.png'), dpi=2048)
    fig.savefig(os.path.join(destination_directory, f'{filename}.png'))
    plt.close(fig)


