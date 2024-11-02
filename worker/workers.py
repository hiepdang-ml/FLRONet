import os
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Literal
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.amp import autocast, GradScaler

from common.training import Accumulator, EarlyStopping, Timer, Logger, CheckpointSaver
from cfd.dataset import CFDDataset, DatasetMixin
from cfd.embedding import Voronoi, Mask
from model import FLRONetWithFNO, FLRONetWithUNet
from common.plotting import plot_frame
from common.functional import compute_velocity_field


class Worker:

    def __init__(self):
        raise ValueError('Base Worker class is not meant to be instantiated')

    #helper
    @staticmethod
    def _validate_inputs(
        sensor_timeframes: torch.Tensor, sensor_frames: torch.Tensor,
        fullstate_timeframes: torch.Tensor, fullstate_frames: torch.Tensor, 
    ) -> Tuple[int, int, int, int, int, int]:
        assert sensor_frames.ndim == fullstate_frames.ndim == 5
        assert sensor_timeframes.ndim == fullstate_timeframes.ndim == 2
        n_sensor_frames: int = sensor_timeframes.shape[1]
        n_fullstate_frames: int = fullstate_timeframes.shape[1]
        assert sensor_frames.shape[0] == fullstate_frames.shape[0]
        batch_size: int = sensor_frames.shape[0]
        n_channels, H, W = sensor_frames.shape[-3:]
        assert sensor_frames.shape == (batch_size, n_sensor_frames, n_channels, H, W)
        assert fullstate_frames.shape == (batch_size, n_fullstate_frames, n_channels, H, W)
        return batch_size, n_sensor_frames, n_fullstate_frames, n_channels, H, W


class Trainer(Worker):

    def __init__(
        self, 
        net: FLRONetWithFNO,
        lr: float,
        train_dataset: CFDDataset,
        val_dataset: CFDDataset,
        train_batch_size: int,
        val_batch_size: int,
    ):
        self.lr: float = lr
        self.train_dataset: CFDDataset = train_dataset
        self.val_dataset: CFDDataset = val_dataset
        self.train_batch_size: int = train_batch_size
        self.val_batch_size: int = val_batch_size

        self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)
        self.val_dataloader = DataLoader(dataset=val_dataset, batch_size=val_batch_size, shuffle=False)
        self.loss_function: nn.Module = nn.MSELoss(reduction='sum')

        self.grad_scaler = GradScaler(device="cuda")
        if torch.cuda.device_count() > 1:
            self.net: FLRONetWithFNO | FLRONetWithUNet = nn.DataParallel(net).cuda()
        elif torch.cuda.device_count() == 1:
            self.net: FLRONetWithFNO | FLRONetWithUNet = net.cuda()
        else:
            raise ValueError('No GPUs are found in the system')
        
        self.optimizer = Adam(params=self.net.parameters(), lr=lr)

    def train(
        self, 
        n_epochs: int,
        patience: int,
        tolerance: float,
        checkpoint_path: Optional[str] = None,
        save_frequency: int = 5,
    ) -> None:
        
        train_metrics = Accumulator()
        early_stopping = EarlyStopping(patience, tolerance)
        timer = Timer()
        logger = Logger()
        checkpoint_saver = CheckpointSaver(
            model=self.net,
            optimizer=self.optimizer,
            dirpath=checkpoint_path,
        )
        checkpoint_prefix: str = 'flronetfno' if isinstance(self.net, FLRONetWithFNO) else 'flronetunet'
        self.net.train()
        
        for epoch in range(1, n_epochs + 1):
            timer.start_epoch(epoch)
            for batch, (
                sensor_timeframes, sensor_frames, fullstate_timeframes, fullstate_frames, _, _
            ) in enumerate(tqdm(self.train_dataloader, desc=f'Epoch {epoch}/{n_epochs}: '), start=1):
                timer.start_batch(epoch, batch)
                # Data validation
                self._validate_inputs(sensor_timeframes, sensor_frames, fullstate_timeframes, fullstate_frames)
                self.optimizer.zero_grad()
                # Use automatic mixed precision to speed up on A100/H100 GPUs
                with autocast(device_type="cuda", dtype=torch.float16):
                    # Forward propagation
                    if isinstance(self.net, FLRONetWithUNet):
                        reconstruction_frames: torch.Tensor = self.net(
                            sensor_timeframes=sensor_timeframes,
                            sensor_values=sensor_frames,
                            fullstate_timeframes=fullstate_timeframes,
                        )
                    else:
                        reconstruction_frames: torch.Tensor = self.net(
                            sensor_timeframes=sensor_timeframes,
                            sensor_values=sensor_frames,
                            fullstate_timeframes=fullstate_timeframes,
                            out_resolution=self.train_dataset.resolution,
                        )
                    # Compute loss
                    total_mse: torch.Tensor = self.loss_function(input=reconstruction_frames, target=fullstate_frames)
                    mean_mse: torch.Tensor = total_mse / reconstruction_frames.numel()

                # Backpropagation
                self.grad_scaler.scale(mean_mse).backward()
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()

                # Accumulate the metrics
                train_metrics.add(total_mse=total_mse.item(), n_elems=reconstruction_frames.numel())
                timer.end_batch(epoch=epoch)
                # Log
                train_mean_mse: float = train_metrics['total_mse'] / train_metrics['n_elems']
                logger.log(
                    epoch=epoch, n_epochs=n_epochs, 
                    batch=batch, n_batches=len(self.train_dataloader), took=timer.time_batch(epoch, batch), 
                    mean_train_rmse=train_mean_mse ** 0.5, mean_train_mse=train_mean_mse,
                )

            # Ragularly save checkpoint
            if checkpoint_path is not None and epoch % save_frequency == 0:
                checkpoint_saver.save(
                    model_states=self.net.state_dict(), 
                    optimizer_states=self.optimizer.state_dict(),
                    filename=f'{checkpoint_prefix}{epoch}.pt',
                )

            # Reset metric records for next epoch
            train_metrics.reset()
            # Evaluate
            mean_val_mse: float = self.evaluate()
            mean_val_rmse: float = mean_val_mse ** 0.5
            timer.end_epoch(epoch)
            # Log
            logger.log(
                epoch=epoch, n_epochs=n_epochs, took=timer.time_epoch(epoch), 
                mean_val_rmse=mean_val_rmse, mean_val_mse=mean_val_mse,
            )
            print('=' * 20)

            # Check early-stopping
            early_stopping(value=mean_val_rmse)
            if early_stopping:
                print('Early Stopped')
                break

        # Always save last checkpoint
        if checkpoint_path:
            checkpoint_saver.save(
                model_states=self.net.state_dict(), 
                optimizer_states=self.optimizer.state_dict(),
                filename=f'{checkpoint_prefix}{epoch}.pt',
            )

    def evaluate(self) -> float:
        val_metrics = Accumulator()
        self.net.eval()
        with torch.no_grad():
            for sensor_timeframes, sensor_frames, fullstate_timeframes, fullstate_frames, _, _ in self.val_dataloader:
                # Data validation
                self._validate_inputs(sensor_timeframes, sensor_frames, fullstate_timeframes, fullstate_frames)
                # Forward propagation
                with autocast(device_type="cuda", dtype=torch.float16):
                    # Forward propagation
                    if isinstance(self.net, FLRONetWithUNet):
                        reconstruction_frames: torch.Tensor = self.net(
                            sensor_timeframes=sensor_timeframes,
                            sensor_values=sensor_frames,
                            fullstate_timeframes=fullstate_timeframes,
                        )
                    else:
                        reconstruction_frames: torch.Tensor = self.net(
                            sensor_timeframes=sensor_timeframes,
                            sensor_values=sensor_frames,
                            fullstate_timeframes=fullstate_timeframes,
                            out_resolution=self.train_dataset.resolution,
                        )
                    # Compute total loss
                    total_mse: torch.Tensor = self.loss_function(input=reconstruction_frames, target=fullstate_frames)

                # Accumulate the val_metrics
                val_metrics.add(total_mse=total_mse.item(), n_elems=reconstruction_frames.numel())

        # Compute the aggregate metrics
        mean_mse: float = val_metrics['total_mse'] / val_metrics['n_elems']
        return mean_mse


class Predictor(Worker, DatasetMixin):

    def __init__(
        self, 
        net: FLRONetWithFNO | FLRONetWithUNet, 

    ):
        self.net = net.cuda()
        self.rmse = nn.MSELoss(reduction='sum')
        self.mae = nn.L1Loss(reduction='sum')

    def predict_from_scratch(
        self, 
        case_dir: str, 
        sensor_timeframes: List[int],
        reconstruction_timeframes: List[int],
        sensor_position_path: str, 
        embedding_generator: Literal['Voronoi', 'Mask'],
        n_dropout_sensors: int,
        in_resolution: Tuple[int, int],
        out_resolution: Tuple[int, int] | None = None,
    ):
        assert isinstance(sensor_position_path, str)
        assert embedding_generator in ('Voronoi', 'Mask')
        assert min(sensor_timeframes) <= min(reconstruction_timeframes)
        assert max(reconstruction_timeframes) <= max(sensor_timeframes)
        n_sensor_timeframes: int = len(sensor_timeframes)
        n_fullstate_timeframes: int = len(reconstruction_timeframes)
        in_H, in_W = in_resolution
        
        # prepare reconstruction timeframes
        reconstruction_timeframes: torch.Tensor = torch.tensor(reconstruction_timeframes, dtype=torch.float, device='cuda')
        reconstruction_timeframes = reconstruction_timeframes.unsqueeze(dim=0)
        # prepare sensor timeframes
        sensor_timeframes: torch.Tensor = torch.tensor(sensor_timeframes, dtype=torch.int, device='cuda')
        sensor_timeframes = sensor_timeframes.unsqueeze(dim=0)
        # load raw data
        data: torch.Tensor = self.load2tensor(case_dir) # already in GPU
        # resize (original resolution is 64 x 64, which is not proportional to 0.14m x 0.24m)
        sensor_frames: torch.Tensor = data[sensor_timeframes]
        sensor_frames = F.interpolate(input=sensor_frames.flatten(0, 1), size=in_resolution, mode='bicubic')
        sensor_frames = sensor_frames.unsqueeze(0)
        # prepare sensor frames
        original_sensor_positions: torch.Tensor = torch.load(sensor_position_path, weights_only=True, map_location='cuda').int()
        if n_dropout_sensors == 0:
            implied_dropout_probabilities: List[float] = []
        else:
            implied_dropout_probabilities: List[float] = [0.] * n_dropout_sensors
            implied_dropout_probabilities[-1] = 1.

        if embedding_generator == 'Mask':
            embedding_generator = Mask(
                resolution=in_resolution, sensor_positions=original_sensor_positions, dropout_probabilities=implied_dropout_probabilities
            )
        else:
            embedding_generator = Voronoi(
                resolution=in_resolution, sensor_positions=original_sensor_positions, dropout_probabilities=implied_dropout_probabilities
            )

        sensor_frames = embedding_generator(data=sensor_frames)
        n_sensors: int = embedding_generator.S
        n_active_sensors: int = n_sensors - n_dropout_sensors

        self.net.eval()
        with torch.no_grad():
            # reconstruct
            with autocast(device_type="cuda", dtype=torch.float16):
                # Forward propagation
                if isinstance(self.net, FLRONetWithUNet):
                    reconstruction_frames: torch.Tensor = self.net(
                        sensor_timeframes=sensor_timeframes,
                        sensor_values=sensor_frames,
                        fullstate_timeframes=reconstruction_timeframes,
                    )
                    assert reconstruction_frames.shape == (
                        1, n_fullstate_timeframes, self.net.n_channels, in_H, in_W
                    )
                else:
                    reconstruction_frames: torch.Tensor = self.net(
                        sensor_timeframes=sensor_timeframes,
                        sensor_values=sensor_frames,
                        fullstate_timeframes=reconstruction_timeframes,
                        out_resolution=out_resolution,
                    )
                    out_H, out_W = out_resolution
                    assert reconstruction_frames.shape == (
                        1, n_fullstate_timeframes, self.net.n_channels, out_H, out_W
                    )

        # visualization
        reconstruction_frames = reconstruction_frames.squeeze(dim=0)
        reconstruction_timeframes = reconstruction_timeframes.squeeze(dim=0)
        case_name: str = os.path.basename(case_dir)
        for frame_idx in tqdm(range(reconstruction_frames.shape[0]), desc=f'{case_name}: '):
            reconstruction_frame: torch.Tensor = reconstruction_frames[frame_idx]
            at_timeframe = float(reconstruction_timeframes[frame_idx].item())
            if isinstance(self.net, FLRONetWithUNet):
                plot_frame(
                    sensor_positions=original_sensor_positions,
                    reconstruction_frame=reconstruction_frame,
                    reduction=lambda x: compute_velocity_field(x, dim=0),
                    title=f'{case_name.upper()} t={at_timeframe * 0.001:.6f}s. active sensors: {str(n_active_sensors).zfill(2)}/{str(n_sensors).zfill(2)}',
                    filename=f'{case_name.lower()}_f{str(at_timeframe).replace(".","")}_{in_H}x{in_W}',
                )
            else:
                new_sensor_position: torch.Tensor = torch.zeros_like(original_sensor_positions, dtype=torch.float)
                new_sensor_position[:, 0] = original_sensor_positions[:, 0] * out_H / in_H
                new_sensor_position[:, 1] = original_sensor_positions[:, 1] * out_W / in_W
                new_sensor_position = new_sensor_position.int()
                plot_frame(
                    sensor_positions=new_sensor_position,
                    reconstruction_frame=reconstruction_frame,
                    reduction=lambda x: compute_velocity_field(x, dim=0),
                    title=f'{case_name.upper()} t={at_timeframe * 0.001:.6f}s. active sensors: {str(n_active_sensors).zfill(2)}/{str(n_sensors).zfill(2)}',
                    filename=f'{case_name.lower()}_f{str(at_timeframe).replace(".","")}_d{n_dropout_sensors}_{out_H}x{out_W}'
                )


    def predict_from_dataset(self, dataset: CFDDataset) -> None:
        self.net.eval()
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        trained_H, trained_W = dataset.resolution
        n_sensors: int = dataset.sensor_positions.shape[0]
        n_dropout_sensors: int = len(dataset.dropout_probabilities)
        n_active_sensors: int = n_sensors - n_dropout_sensors
        rmse_values: List[float] = []
        mae_values: List[float] = []
        with torch.no_grad():
            for sensor_timeframes, sensor_frames, fullstate_timeframes, fullstate_frames, case_names, sampling_ids in tqdm(dataloader):
                # Data validation
                self._validate_inputs(sensor_timeframes, sensor_frames, fullstate_timeframes, fullstate_frames)
                # Forward propagation
                with autocast(device_type="cuda", dtype=torch.float16):
                    # Forward propagation
                    if isinstance(self.net, FLRONetWithUNet):
                        reconstruction_frames: torch.Tensor = self.net(
                            sensor_timeframes=sensor_timeframes,
                            sensor_values=sensor_frames,
                            fullstate_timeframes=fullstate_timeframes,
                        )
                    else:
                        reconstruction_frames: torch.Tensor = self.net(
                            sensor_timeframes=sensor_timeframes,
                            sensor_values=sensor_frames,
                            fullstate_timeframes=fullstate_timeframes,
                            out_resolution=dataset.resolution,
                        )

                # Visualization
                reconstruction_frames = reconstruction_frames.squeeze(dim=0)
                fullstate_frames = fullstate_frames.squeeze(dim=0)
                fullstate_timeframes = fullstate_timeframes.squeeze(dim=0)
                for frame_idx in range(fullstate_timeframes.shape[0]):
                    reconstruction_frame: torch.Tensor = reconstruction_frames[frame_idx]
                    fullstate_frame: torch.Tensor = fullstate_frames[frame_idx]
                    frame_total_mse: torch.Tensor = self.rmse(
                        input=reconstruction_frame.unsqueeze(0).unsqueeze(0), 
                        target=fullstate_frame.unsqueeze(0).unsqueeze(0),
                    )
                    frame_mean_mse: float = frame_total_mse.item() / fullstate_frame.numel()
                    frame_mean_rmse: float = frame_mean_mse ** 0.5
                    frame_total_mae: torch.Tensro = self.mae(
                        input=reconstruction_frame.unsqueeze(0).unsqueeze(0), 
                        target=fullstate_frame.unsqueeze(0).unsqueeze(0),
                    )
                    frame_mean_mae: float = frame_total_mae.item() / fullstate_frame.numel()
                    at_timeframe = int(fullstate_timeframes[frame_idx].item())
                    case_name: str = case_names[frame_idx]
                    sampling_id: int = sampling_ids[frame_idx]
                    plot_frame(
                        sensor_positions=dataset.sensor_positions,
                        fullstate_frame=fullstate_frame, 
                        reconstruction_frame=reconstruction_frame,
                        reduction=lambda x: compute_velocity_field(x, dim=0),
                        title=(
                            f'{case_name.lower()}s{sampling_id}: '
                            f't={at_timeframe * 0.001:.3f}s, '
                            f'active sensors: {str(n_active_sensors).zfill(2)}/{str(n_sensors).zfill(2)}, '
                            f'RMSE: {frame_mean_rmse:.3f}, MAE: {frame_mean_mae:.3f}'
                        ),
                        filename=f'{case_name.lower()}s{sampling_id}_f{str(at_timeframe).zfill(3)}_d{n_dropout_sensors}_{trained_H}x{trained_W}'
                    )
                    rmse_values.append(frame_mean_rmse)
                    mae_values.append(frame_mean_mae)

        return sum(rmse_values) / len(rmse_values), sum(mae_values) / len(mae_values)
