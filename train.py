import argparse
from typing import List, Dict, Any, Optional

import yaml
from torch.optim import Optimizer, Adam

from sensors import LHS, AroundCylinder
from embeddings import Mask, Voronoi
from models import FLRONet
from datasets import CFDDataset
from common.training import CheckpointLoader
from workers import Trainer


def main(config: Dict[str, Any]) -> None:
    """
    Main function to train FLRONet.

    Parameters:
        config (Dict[str, Any]): Configuration dictionary.
    """

    # Parse CLI arguments:
    init_sensor_timeframe_indices: List[int]    = list(config['dataset']['init_sensor_timeframe_indices'])
    n_fullstate_timeframes_per_chunk: int       = int(config['dataset']['n_fullstate_timeframes_per_chunk'])
    n_samplings_per_chunk: int                  = int(config['dataset']['n_samplings_per_chunk'])
    resolution: tuple                           = tuple(config['dataset']['resolution'])
    n_sensors: int                              = int(config['dataset']['n_sensors'])
    sensor_generator: str                       = str(config['dataset']['sensor_generator'])
    embedding_generator: str                    = str(config['dataset']['embedding_generator'])
    seed: int                                   = int(config['dataset']['seed'])

    n_channels: int                             = int(config['architecture']['n_channels'])
    n_afno_layers: int                          = int(config['architecture']['n_afno_layers'])
    embedding_dim: int                          = int(config['architecture']['embedding_dim'])
    block_size: int                             = int(config['architecture']['block_size'])
    dropout_rate: float                         = float(config['architecture']['dropout_rate'])
    n_stacked_networks: int                     = int(config['architecture']['n_stacked_networks'])
    from_checkpoint: Optional[str]              = config['architecture']['from_checkpoint']
    
    train_batch_size: int                       = int(config['training']['train_batch_size'])
    val_batch_size: int                         = int(config['training']['val_batch_size'])
    learning_rate: float                        = float(config['training']['learning_rate'])
    a: float                                    = float(config['training']['a'])
    r: float                                    = float(config['training']['r'])
    n_epochs: int                               = int(config['training']['n_epochs'])
    patience: int                               = int(config['training']['patience'])
    tolerance: int                              = float(config['training']['tolerance'])
    save_frequency: int                         = int(config['training']['save_frequency'])

    # Instatiate the sensor generator
    if sensor_generator == 'AroundCylinder':
        sensor_generator = AroundCylinder(resolution, n_sensors)
    elif sensor_generator == 'LHS':
        sensor_generator = LHS(resolution, n_sensors)
    else:
        raise ValueError(f'Invalid sensor_generator: {sensor_generator}')

    # Instatiate the embedding generator
    if embedding_generator == 'Mask':
        embedding_generator = Mask()
    elif embedding_generator == 'Voronoi':
        embedding_generator = Voronoi()
    else:
        raise ValueError(f'Invalid embedding_generator: {embedding_generator}')

    # Instatiate the training datasets
    train_dataset = CFDDataset(
        root='./data/train', 
        init_sensor_timeframe_indices=init_sensor_timeframe_indices,
        n_fullstate_timeframes_per_chunk=n_fullstate_timeframes_per_chunk,
        n_samplings_per_chunk=n_samplings_per_chunk,
        resolution=resolution,
        sensor_generator=sensor_generator, 
        embedding_generator=embedding_generator,
        seed=seed,
    )
    val_dataset = CFDDataset(
        root='./data/val', 
        init_sensor_timeframe_indices=init_sensor_timeframe_indices,
        n_fullstate_timeframes_per_chunk=n_fullstate_timeframes_per_chunk,
        n_samplings_per_chunk=n_samplings_per_chunk,
        resolution=resolution,
        sensor_generator=sensor_generator, 
        embedding_generator=embedding_generator,
        seed=seed,
    )

    # Load the model
    if from_checkpoint is not None:
        checkpoint_loader = CheckpointLoader(checkpoint_path=from_checkpoint)
        net: FLRONet = checkpoint_loader.load(scope=globals())[0].cuda()    # ignore optimizer
    else:
        net = FLRONet(
            n_channels=n_channels, n_afno_layers=n_afno_layers, 
            embedding_dim=embedding_dim, block_size=block_size, 
            dropout_rate=dropout_rate, 
            n_fullstate_timeframes=n_fullstate_timeframes_per_chunk, 
            n_sensor_timeframes=len(init_sensor_timeframe_indices), 
            resolution=resolution,
            n_stacked_networks=n_stacked_networks,
        ).cuda()
        
    # Load global trainer    
    trainer = Trainer(
        net=net, 
        lr=learning_rate,
        a=a, r=r,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
    )
    trainer.train(
        n_epochs=n_epochs, 
        patience=patience,
        tolerance=tolerance, 
        checkpoint_path=f'.checkpoints',
        save_frequency=save_frequency,
    )


if __name__ == "__main__":

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Train FLRONet')
    parser.add_argument('--config', type=str, required=True, help='Configuration file name.')

    args: argparse.Namespace = parser.parse_args()
    
    # Load the configuration
    with open(file=args.config, mode='r') as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    # Run the main function with the configuration
    main(config)


