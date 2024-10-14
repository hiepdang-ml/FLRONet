from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from functools import cached_property

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import autocast, GradScaler

from common.training import Accumulator, EarlyStopping, Timer, Logger, CheckpointSaver
from datasets import CFDDataset
from models import FLRONet
from losses import CustomMSE
from plotting import plot_frame
from functional import compute_velocity_field


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
        n_sensor_frames: int = sensor_timeframes.shape[0]
        n_fullstate_frames: int = fullstate_timeframes.shape[0]
        assert sensor_timeframes.shape[1] == fullstate_timeframes.shape[1]
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
            sensor_positions=train_dataset.sensor_positions,
            a=a, r=r,
            reduction='sum',
        ).cuda()

        self.optimizer: AdamW = AdamW(params=net.parameters, lr=lr)

        self.grad_scaler = GradScaler(device="cuda")
        if torch.cuda.device_count() > 1:
            self.net: FLRONet = nn.DataParallel(net).cuda()
        elif torch.cuda.device_count() == 1:
            self.net: FLRONet = net.cuda()
        else:
            raise ValueError('No GPUs are found in the system')

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
            model=self.global_operator,
            optimizer=self.optimizer,
            dirpath=checkpoint_path,
        )
        self.net.train()
        
        for epoch in range(1, n_epochs + 1):
            timer.start_epoch(epoch)
            for batch, (
                sensor_timeframes, sensor_frames, fullstate_timeframes, fullstate_frames
            ) in enumerate(self.train_dataloader, start=1):
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
                        sensor_timeframe_tensor=sensor_timeframes,
                        sensor_tensor=sensor_frames,
                        fullstate_timeframe_tensor=fullstate_timeframes,
                    )
                    # Compute loss
                    total_loss: torch.Tensor = self.loss_function(
                        fullstate_frames=fullstate_frames, reconstructed_frames=reconstruction_frames
                    )
                    mean_loss: torch.Tensor = total_loss / reconstruction_frames.numel()
            
                # Backpropagation
                self.grad_scaler.scale(mean_loss).backward()
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()

                # Accumulate the metrics
                train_metrics.add(total_loss=total_loss.item(), n_elems=reconstruction_frames.numel())
                timer.end_batch(epoch=epoch)
                # Log
                train_mean_loss: float = train_metrics['total_loss'] / train_metrics['n_elems']
                logger.log(
                    epoch=epoch, n_epochs=n_epochs, 
                    batch=batch, n_batches=len(self.train_dataloader), 
                    took=timer.time_batch(epoch, batch), 
                    mean_train_loss=train_mean_loss
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
            val_mean_loss = self.evaluate()
            timer.end_epoch(epoch)
            # Log
            logger.log(
                epoch=epoch, n_epochs=n_epochs, 
                took=timer.time_epoch(epoch), 
                val_mean_loss=val_mean_loss, 
            )
            print('=' * 20)

            # Check early-stopping
            early_stopping(value=val_mean_loss)
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
            for sensor_timeframes, sensor_frames, fullstate_timeframes, fullstate_frames in self.val_dataloader:
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
                        sensor_timeframe_tensor=sensor_timeframes,
                        sensor_tensor=sensor_frames,
                        fullstate_timeframe_tensor=fullstate_timeframes,
                    )
                    # Compute total loss
                    total_loss: torch.Tensor = self.loss_function(
                        fullstate_frames=fullstate_frames, reconstructed_frames=reconstruction_frames
                    )

                # Accumulate the val_metrics
                val_metrics.add(total_loss=total_loss.item(), n_elems=reconstruction_frames.numel())

        # Compute the aggregate metrics
        mean_loss: float = val_metrics['total_loss'] / val_metrics['n_elems']
        return mean_loss


class Predictor(Worker):

    def __init__(self, net: FLRONet):
        self.net = net.cuda()
        self.loss_function: nn.Module = nn.MSELoss(reduction='sum').to(device=self.device)

    def predict(self, dataset: CFDDataset) -> None:
        self.net.eval()
        # Batch size should be 1 since len(dataset) == 1
        dataloader = DataLoader(
            dataset, 
            batch_size=1, 
            num_workers=4, 
            prefetch_factor=3, 
            pin_memory=True,
            shuffle=False
        )
        with torch.no_grad():
            for sensor_timeframes, sensor_frames, fullstate_timeframes, fullstate_frames in dataloader:
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
                        sensor_timeframe_tensor=sensor_timeframes,
                        sensor_tensor=sensor_frames,
                        fullstate_timeframe_tensor=fullstate_timeframes,
                    )

                # Visualization
                reconstruction_frames = reconstruction_frames.squeeze(dim=0)
                fullstate_frames = fullstate_frames.squeeze(dim=0)
                fullstate_timeframes = fullstate_timeframes.squeeze(dim=0)

                for frame_idx in range(fullstate_timeframes.shape[0]):
                    at_timeframe = int(fullstate_timeframes[frame_idx].item())
                    plot_frame(
                        sensor_positions=dataset.sensor_positions,
                        fullstate_frame=fullstate_frames[frame_idx], 
                        reconstruction_frame=reconstruction_frames[frame_idx],
                        reduction=lambda x: compute_velocity_field(x, dim=0),
                        sufix=f"at t={at_timeframe * 0.001}s (frame {at_timeframe})"
                    )

