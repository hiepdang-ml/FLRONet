import os
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
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
from model import FLRONet
from model.losses import CustomMSE
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

    #helper
    @staticmethod
    def _move2gpu(*tensors) -> Tuple[torch.Tensor,... ]:
        return tuple(tensor.cuda() for tensor in tensors)

class Trainer(Worker):

    def __init__(
        self, 
        net: FLRONet,
        lr: float,
        a: float,
        r: float,
        train_dataset: CFDDataset,
        val_dataset: CFDDataset,
        train_batch_size: int,
        val_batch_size: int,
    ):
        self.lr: float = lr
        self.a: float = a
        self.r: float = r
        self.train_dataset: CFDDataset = train_dataset
        self.val_dataset: CFDDataset = val_dataset
        self.train_batch_size: int = train_batch_size
        self.val_batch_size: int = val_batch_size

        self.train_dataloader = DataLoader(
            dataset=train_dataset, 
            batch_size=train_batch_size, 
            shuffle=True,
            num_workers=4,
            prefetch_factor=3,
            pin_memory=True,
        )
        self.val_dataloader = DataLoader(
            dataset=val_dataset, 
            batch_size=val_batch_size, 
            shuffle=False,
            num_workers=4,
            prefetch_factor=3,
            pin_memory=True,
        )
        self.loss_function: nn.Module = CustomMSE(
            resolution=train_dataset.resolution,
            sensor_positions=train_dataset.sensor_positions.cuda(),
            a=a, r=r,
            reduction='sum',
        )
        self.metric = nn.MSELoss(reduction='sum')

        self.grad_scaler = GradScaler(device="cuda")
        if torch.cuda.device_count() > 1:
            self.net: FLRONet = nn.DataParallel(net).cuda()
        elif torch.cuda.device_count() == 1:
            self.net: FLRONet = net.cuda()
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
        self.net.train()
        
        for epoch in range(1, n_epochs + 1):
            timer.start_epoch(epoch)
            # in enumerate(tqdm(self.train_dataloader, start=1), start=1):
            for batch, (
                sensor_timeframes, sensor_frames, fullstate_timeframes, fullstate_frames, _, _
            ) in enumerate(tqdm(self.train_dataloader, desc=f'Epoch {epoch}/{n_epochs}: '), start=1):
                timer.start_batch(epoch, batch)
                # Data validation
                self._validate_inputs(sensor_timeframes, sensor_frames, fullstate_timeframes, fullstate_frames)
                # Move to GPU
                sensor_timeframes, sensor_frames, fullstate_timeframes, fullstate_frames = self._move2gpu(
                    sensor_timeframes, sensor_frames, fullstate_timeframes, fullstate_frames
                )
                self.optimizer.zero_grad()
                # Use automatic mixed precision to speed up on A100/H100 GPUs
                with autocast(device_type="cuda", dtype=torch.float16):
                    # Forward propagation
                    reconstruction_frames: torch.Tensor = self.net(
                        sensor_timeframes=sensor_timeframes,
                        sensor_values=sensor_frames,
                        fullstate_timeframes=fullstate_timeframes,
                    )
                    # Compute loss
                    total_loss: torch.Tensor = self.loss_function(
                        reconstructed_frames=reconstruction_frames, fullstate_frames=fullstate_frames
                    )
                    mean_loss: torch.Tensor = total_loss / reconstruction_frames.numel()
                    # Compute metrics
                    with torch.no_grad():
                        total_mse: torch.Tensor = self.metric(
                            input=reconstruction_frames, target=fullstate_frames,
                        )

                # Backpropagation
                self.grad_scaler.scale(mean_loss).backward()
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()

                # Accumulate the metrics
                train_metrics.add(
                    total_loss=total_loss.item(), 
                    total_mse=total_mse.item(),
                    n_elems=reconstruction_frames.numel(),
                )
                timer.end_batch(epoch=epoch)
                # Log
                train_mean_loss: float = train_metrics['total_loss'] / train_metrics['n_elems']
                train_mean_mse: float = train_metrics['total_mse'] / train_metrics['n_elems']
                logger.log(
                    epoch=epoch, n_epochs=n_epochs, 
                    batch=batch, n_batches=len(self.train_dataloader), took=timer.time_batch(epoch, batch), 
                    mean_train_rmse=train_mean_mse ** 0.5, mean_train_mse=train_mean_mse, mean_train_loss=train_mean_loss
                )
        
            # Ragularly save checkpoint
            if checkpoint_path is not None and epoch % save_frequency == 0:
                checkpoint_saver.save(
                    model_states=self.net.state_dict(), 
                    optimizer_states=self.optimizer.state_dict(),
                    filename=f'epoch{epoch}.pt',
                )
            
            # Reset metric records for next epoch
            train_metrics.reset()
            # Evaluate
            val_mean_mse: float; val_mean_rmse: float; val_mean_loss: float
            val_mean_mse, val_mean_loss = self.evaluate()
            val_mean_rmse = val_mean_mse ** 0.5
            timer.end_epoch(epoch)
            # Log
            logger.log(
                epoch=epoch, n_epochs=n_epochs, took=timer.time_epoch(epoch), 
                val_mean_rmse=val_mean_rmse, val_mean_mse=val_mean_mse, val_mean_loss=val_mean_loss, 
            )
            print('=' * 20)

            # Check early-stopping
            early_stopping(value=val_mean_rmse)
            if early_stopping:
                print('Early Stopped')
                break

        # Always save last checkpoint
        if checkpoint_path:
            checkpoint_saver.save(
                model_states=self.global_operator.state_dict(), 
                optimizer_states=self.optimizer.state_dict(),
                filename=f'epoch{epoch}.pt',
            )

    def evaluate(self) -> float:
        val_metrics = Accumulator()
        self.net.eval()
        with torch.no_grad():
            for sensor_timeframes, sensor_frames, fullstate_timeframes, fullstate_frames, _, _ in self.val_dataloader:
                # Data validation
                self._validate_inputs(sensor_timeframes, sensor_frames, fullstate_timeframes, fullstate_frames)
                # Move to GPU
                sensor_timeframes, sensor_frames, fullstate_timeframes, fullstate_frames = self._move2gpu(
                    sensor_timeframes, sensor_frames, fullstate_timeframes, fullstate_frames
                )
                # Forward propagation
                with autocast(device_type="cuda", dtype=torch.float16):
                    # Forward propagation
                    reconstruction_frames: torch.Tensor = self.net(
                        sensor_timeframes=sensor_timeframes,
                        sensor_values=sensor_frames,
                        fullstate_timeframes=fullstate_timeframes,
                    )
                    # Compute total loss
                    total_loss: torch.Tensor = self.loss_function(
                        reconstructed_frames=reconstruction_frames, fullstate_frames=fullstate_frames,
                    )
                    # Compute metrics
                    total_mse: torch.Tensor = self.metric(
                        input=reconstruction_frames, target=fullstate_frames,
                    )

                # Accumulate the val_metrics
                val_metrics.add(
                    total_loss=total_loss.item(), 
                    total_mse=total_mse.item(), 
                    n_elems=reconstruction_frames.numel(), 
                )

        # Compute the aggregate metrics
        mean_mse: float = val_metrics['total_mse'] / val_metrics['n_elems']
        mean_loss: float = val_metrics['total_loss'] / val_metrics['n_elems']
        return mean_mse, mean_loss


class Predictor(Worker, DatasetMixin):

    def __init__(
        self, 
        net: FLRONet, 
        sensor_position_path: str | None = None, 
        embedding_generator: Voronoi | Mask | None = None
    ):
        self.net = net.cuda()
        self.sensor_position_path: str | None = sensor_position_path
        self.embedding_generator: Voronoi | Mask | None = embedding_generator
        if sensor_position_path is not None:
            self.sensor_positions: torch.Tensor = torch.load(sensor_position_path).cuda()
    
        self.H, self.W = self.net.resolution
        self.loss_function: nn.Module = nn.MSELoss(reduction='sum')
        self.metric = nn.MSELoss(reduction='sum')

    def predict_from_scratch(
        self, 
        case_dir: str, 
        sensor_timeframes: List[int],
        reconstruction_timeframes: List[int],
    ):
        assert isinstance(self.sensor_position_path, str)
        assert isinstance(self.embedding_generator, (Voronoi, Mask))
        assert len(sensor_timeframes) == self.net.n_sensor_timeframes
        assert len(reconstruction_timeframes) == self.net.n_fullstate_timeframes
        assert min(sensor_timeframes) < min(reconstruction_timeframes)
        assert max(reconstruction_timeframes) < max(sensor_timeframes)

        self.net.eval()
        # load raw data
        data: torch.Tensor = self.load2tensor(case_dir).cuda()
        # prepare reconstruction timeframes
        reconstruction_timeframes: torch.Tensor = torch.tensor(reconstruction_timeframes, dtype=torch.long).cuda()
        reconstruction_timeframes = reconstruction_timeframes.unsqueeze(dim=0)
        # prepare sensor timeframes
        sensor_timeframes: torch.Tensor = torch.tensor(sensor_timeframes, dtype=torch.long).cuda()
        sensor_timeframes = sensor_timeframes.unsqueeze(dim=0)
        # prepare sensor values
        sensor_frames: torch.Tensor = data[sensor_timeframes]
        # resize sensor frames (original resolution is 64 x 64, which is not proportional to 0.14m x 0.24m)
        sensor_frames = F.interpolate(
            input=sensor_frames.flatten(0, 1), size=(self.H, self.W), mode='bicubic'
        )
        sensor_frames = sensor_frames.reshape(1, self.net.n_sensor_timeframes, 2, self.H, self.W)
        # compute sensor data for entire space
        sensor_frames = self.embedding_generator(data=sensor_frames, sensor_positions=self.sensor_positions)
        assert sensor_frames.shape == (1, self.net.n_sensor_timeframes, 2, self.H, self.W)
        with torch.no_grad():
            # reconstruct
            with autocast(device_type="cuda", dtype=torch.float16):
                # Forward propagation
                reconstruction_frames: torch.Tensor = self.net(
                    sensor_timeframes=sensor_timeframes,
                    sensor_values=sensor_frames,
                    fullstate_timeframes=reconstruction_timeframes,
                )

        assert reconstruction_frames.shape == (
            1, len(reconstruction_timeframes), self.n_channels, self.H, self.W
        )
        # visualization
        reconstruction_frames = reconstruction_frames.squeeze(dim=0)
        reconstruction_timeframes = reconstruction_timeframes.squeeze(dim=0)
        case_name: str = os.path.basename(case_dir)
        for frame_idx in range(reconstruction_frames.shape[0]):
            reconstruction_frame: torch.Tensor = reconstruction_frames[frame_idx]
            at_timeframe = int(reconstruction_timeframes[frame_idx].item())
            plot_frame(
                sensor_positions=self.sensor_positions,
                reconstruction_frame=reconstruction_frame,
                reduction=lambda x: compute_velocity_field(x, dim=0),
                prefix=f'{case_name.upper()}\n',
                suffix=f"at t={at_timeframe * 0.001}s (frame {at_timeframe})",
            )

    def predict_from_dataset(self, dataset: CFDDataset) -> None:
        self.net.eval()
        dataloader = DataLoader(
            dataset, 
            batch_size=1, 
            num_workers=4, 
            prefetch_factor=3, 
            pin_memory=True,
            shuffle=False
        )
        with torch.no_grad():
            for sensor_timeframes, sensor_frames, fullstate_timeframes, fullstate_frames, case_name, sampling_id in dataloader:
                # Data validation
                self._validate_inputs(sensor_timeframes, sensor_frames, fullstate_timeframes, fullstate_frames)
                # Move to GPU
                sensor_timeframes, sensor_frames, fullstate_timeframes, fullstate_frames = self._move2gpu(
                    sensor_timeframes, sensor_frames, fullstate_timeframes, fullstate_frames
                )
                # Forward propagation
                with autocast(device_type="cuda", dtype=torch.float16):
                    # Forward propagation
                    reconstruction_frames: torch.Tensor = self.net(
                        sensor_timeframes=sensor_timeframes,
                        sensor_values=sensor_frames,
                        fullstate_timeframes=fullstate_timeframes,
                    )

                # Visualization
                reconstruction_frames = reconstruction_frames.squeeze(dim=0)
                fullstate_frames = fullstate_frames.squeeze(dim=0)
                fullstate_timeframes = fullstate_timeframes.squeeze(dim=0)

                for frame_idx in range(fullstate_timeframes.shape[0]):
                    reconstruction_frame: torch.Tensor = reconstruction_frames[frame_idx]
                    fullstate_frame: torch.Tensor = fullstate_frames[frame_idx]
                    frame_total_mse: torch.Tensor = self.metric(
                        input=reconstruction_frame.unsqueeze(0).unsqueeze(0), 
                        target=fullstate_frame.unsqueeze(0).unsqueeze(0),
                    )
                    frame_mean_mse: float = frame_total_mse.item() / fullstate_frame.numel()
                    frame_mean_rmse: float = frame_mean_mse ** 0.5
                    at_timeframe = int(fullstate_timeframes[frame_idx].item())
                    plot_frame(
                        sensor_positions=dataset.sensor_positions,
                        fullstate_frame=fullstate_frame, 
                        reconstruction_frame=reconstruction_frame,
                        reduction=lambda x: compute_velocity_field(x, dim=0),
                        prefix=f'{case_name.upper()}, Samping {sampling_id}\n',
                        suffix=f"at t={at_timeframe * 0.001}s (frame {at_timeframe}) | RMSE: {frame_mean_rmse:.6f}",
                    )
