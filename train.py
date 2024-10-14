import argparse
from typing import Tuple, Dict, Any, Optional

import yaml

import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch.optim import Optimizer, Adam

from models.operators import GlobalOperator
from era5.datasets import ERA5_6Hour
from common.training import CheckpointLoader
from workers.trainer import GlobalOperatorTrainer


def main(config: Dict[str, Any]) -> None:
    """
    Main function to train FLRONet.

    Parameters:
        config (Dict[str, Any]): Configuration dictionary.
    """

    # Parse CLI arguments:
    global_latitude: Tuple[float, float] = tuple(config['dataset']['global_latitude'])
    global_longitude: Tuple[float, float] = tuple(config['dataset']['global_longitude'])
    global_resolution: Tuple[int, int]  = tuple(config['dataset']['global_resolution'])
    train_fromyear: str                 = int(config['dataset']['train_fromyear'])
    train_toyear: str                   = int(config['dataset']['train_toyear'])
    val_fromyear: str                   = int(config['dataset']['val_fromyear'])
    val_toyear: str                     = int(config['dataset']['val_toyear'])
    indays: int                         = int(config['dataset']['indays'])
    outdays: int                        = int(config['dataset']['outdays'])

    embedding_dim: int                  = int(config['global_architecture']['embedding_dim'])
    n_layers: int                       = int(config['global_architecture']['n_layers'])
    block_size: int                     = int(config['global_architecture']['block_size'])
    patch_size: int                     = tuple(config['global_architecture']['patch_size'])
    dropout_rate: float                 = float(config['global_architecture']['dropout_rate'])
    from_checkpoint: Optional[str]      = config['global_architecture']['from_checkpoint']
    
    noise_level: float                  = float(config['training']['noise_level'])
    train_batch_size: int               = int(config['training']['train_batch_size'])
    val_batch_size: int                 = int(config['training']['val_batch_size'])
    learning_rate: float                = float(config['training']['learning_rate'])
    n_epochs: int                       = int(config['training']['n_epochs'])
    patience: int                       = int(config['training']['patience'])
    tolerance: int                      = float(config['training']['tolerance'])
    save_frequency: int                 = int(config['training']['save_frequency'])

    # Instatiate the training datasets
    train_dataset = ERA5_6Hour(
        fromyear=train_fromyear,
        toyear=train_toyear,
        global_latitude=global_latitude,
        global_longitude=global_longitude,
        global_resolution=global_resolution,
        local_latitude=None,
        local_longitude=None,
        indays=indays,
        outdays=outdays,
    )
    val_dataset = ERA5_6Hour(
        fromyear=val_fromyear,
        toyear=val_toyear,
        global_latitude=global_latitude,
        global_longitude=global_longitude,
        global_resolution=global_resolution,
        local_latitude=None,
        local_longitude=None,
        indays=indays,
        outdays=outdays,
    )

    # Load global operator
    if from_checkpoint is not None:
        checkpoint_loader = CheckpointLoader(checkpoint_path=from_checkpoint)
        operator: GlobalOperator = checkpoint_loader.load(scope=globals())[0]   # ignore optimizer
    else:
        operator = GlobalOperator(
            in_channels=train_dataset.in_channels, 
            out_channels=train_dataset.out_channels,
            embedding_dim=embedding_dim,
            in_timesteps=train_dataset.in_timesteps, 
            out_timesteps=train_dataset.out_timesteps,
            n_layers=n_layers,
            spatial_resolution=train_dataset.global_resolution,
            block_size=block_size, 
            patch_size=patch_size,
            dropout_rate=dropout_rate,
        )
        
    optimizer = Adam(params=operator.parameters(), lr=learning_rate)

    # Load global trainer    
    trainer = GlobalOperatorTrainer(
        global_operator=operator, 
        optimizer=optimizer,
        noise_level=noise_level,
        train_dataset=train_dataset, 
        val_dataset=val_dataset,
        train_batch_size=train_batch_size, 
        val_batch_size=val_batch_size,
        device=torch.device('cuda'),
    )
    trainer.train(
        n_epochs=n_epochs, 
        patience=patience,
        tolerance=tolerance, 
        checkpoint_path=f'.checkpoints/global',
        save_frequency=save_frequency,
    )


if __name__ == "__main__":

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Train the Global Operator')
    parser.add_argument('--config', type=str, required=True, help='Configuration file name.')

    args: argparse.Namespace = parser.parse_args()
    
    # Load the configuration
    with open(file=args.config, mode='r') as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    # Run the main function with the configuration
    main(config)


