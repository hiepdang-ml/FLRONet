import argparse
from typing import List, Dict, Any, Optional
import yaml

from model import FLRONetFNO, FLRONetUNet, FLRONetMLP, FNO3D
from cfd.dataset import CFDDataset
from common.training import CheckpointLoader
from worker import Trainer


def main(config: Dict[str, Any]) -> None:
    """
    Main function to train FLRONet.

    Parameters:
        config (Dict[str, Any]): Configuration dictionary.
    """

    # Parse CLI arguments:
    init_sensor_timeframes: List[int]           = list(config['dataset']['init_sensor_timeframes'])
    n_fullstate_timeframes_per_chunk: int       = int(config['dataset']['n_fullstate_timeframes_per_chunk'])
    n_samplings_per_chunk: int                  = int(config['dataset']['n_samplings_per_chunk'])
    resolution: tuple                           = tuple(config['dataset']['resolution'])
    n_sensors: int                              = int(config['dataset']['n_sensors'])
    sensor_generator: str                       = str(config['dataset']['sensor_generator'])
    embedding_generator: str                    = str(config['dataset']['embedding_generator'])
    dropout_probabilities: List[float]          = list(config['dataset']['dropout_probabilities'])
    seed: int                                   = int(config['dataset']['seed'])
    model_name: str                             = str(config['architecture']['momdel_name'])
    n_channels: int                             = int(config['architecture']['n_channels'])
    embedding_dim: int                          = int(config['architecture']['embedding_dim'])
    n_stacked_networks: int                     = int(config['architecture']['n_stacked_networks'])
    n_fno_layers: int                           = int(config['architecture']['n_fno_layers'])
    n_hmodes: int                               = int(config['architecture']['n_hmodes'])
    n_wmodes: int                               = int(config['architecture']['n_wmodes'])
    n_tmodes: int                               = int(config['architecture']['n_tmodes'])
    from_checkpoint: Optional[str]              = config['training']['from_checkpoint']
    train_batch_size: int                       = int(config['training']['train_batch_size'])
    val_batch_size: int                         = int(config['training']['val_batch_size'])
    learning_rate: float                        = float(config['training']['learning_rate'])
    n_epochs: int                               = int(config['training']['n_epochs'])
    patience: int                               = int(config['training']['patience'])
    tolerance: int                              = float(config['training']['tolerance'])
    save_frequency: int                         = int(config['training']['save_frequency'])
    freeze_branchnets: bool                     = bool(config['training']['freeze_branchnets'])
    freeze_trunknets: bool                      = bool(config['training']['freeze_trunknets'])
    freeze_bias: bool                           = bool(config['training']['freeze_bias'])

    # Dataset
    train_dataset = CFDDataset(
        root='./data/train', 
        init_sensor_timeframes=init_sensor_timeframes,
        n_fullstate_timeframes_per_chunk=n_fullstate_timeframes_per_chunk if model_name.lower().startswith('fllronet') else len(init_sensor_timeframes),
        n_samplings_per_chunk=n_samplings_per_chunk,
        resolution=resolution,
        n_sensors=n_sensors,
        dropout_probabilities=dropout_probabilities,
        noise_level=0.,
        sensor_generator=sensor_generator, 
        embedding_generator=embedding_generator,
        init_fullstate_timeframes=None if model_name.lower().startswith('fllronet') else init_sensor_timeframes,
        seed=seed,
    )
    val_dataset = CFDDataset(
        root='./data/test', 
        init_sensor_timeframes=init_sensor_timeframes,
        n_fullstate_timeframes_per_chunk=n_fullstate_timeframes_per_chunk if model_name.lower().startswith('fllronet') else len(init_sensor_timeframes),
        n_samplings_per_chunk=n_samplings_per_chunk,
        resolution=resolution,
        n_sensors=n_sensors,
        dropout_probabilities=dropout_probabilities,
        noise_level=0.,
        sensor_generator=sensor_generator, 
        embedding_generator=embedding_generator,
        init_fullstate_timeframes=None if model_name.lower().startswith('fllronet') else init_sensor_timeframes,
        seed=seed,
    )

    if model_name.lower() == 'flronet-fno':
        # Model
        if from_checkpoint is not None:
            checkpoint_loader = CheckpointLoader(checkpoint_path=from_checkpoint)
            net: FLRONetFNO = checkpoint_loader.load(scope=globals()).cuda()    # ignore optimizer
            assert isinstance(net, FLRONetFNO)
        else:
            net = FLRONetFNO(
                n_channels=n_channels, n_fno_layers=n_fno_layers, 
                n_hmodes=n_hmodes, n_wmodes=n_wmodes, embedding_dim=embedding_dim,
                total_timeframes=train_dataset.total_timeframes_per_case,
                n_stacked_networks=n_stacked_networks,
            ).cuda()

    elif model_name.lower() == 'flronet-unet':
        # Model
        if from_checkpoint is not None:
            checkpoint_loader = CheckpointLoader(checkpoint_path=from_checkpoint)
            net: FLRONetUNet = checkpoint_loader.load(scope=globals()).cuda()
            assert isinstance(net, FLRONetUNet)
        else:
            net = FLRONetUNet(
                n_channels=n_channels, embedding_dim=embedding_dim, n_stacked_networks=n_stacked_networks
            ).cuda()
    
    elif model_name.lower() == 'flronet-mlp':
        # Model
        if from_checkpoint is not None:
            checkpoint_loader = CheckpointLoader(checkpoint_path=from_checkpoint)
            net: FLRONetMLP = checkpoint_loader.load(scope=globals()).cuda()
            assert isinstance(net, FLRONetMLP)
        else:
            net = FLRONetMLP(
                n_channels=n_channels, embedding_dim=embedding_dim, n_sensors=n_sensors, 
                resolution=resolution, n_stacked_networks=n_stacked_networks,
            ).cuda()

    elif model_name.lower() == 'fno3d':
        # Model
        if from_checkpoint is not None:
            checkpoint_loader = CheckpointLoader(checkpoint_path=from_checkpoint)
            net: FNO3D = checkpoint_loader.load(scope=globals()).cuda()
            assert isinstance(net, FNO3D)
        else:
            net = FNO3D(
                n_channels=n_channels, n_fno_layers=n_fno_layers, 
                n_hmodes=n_hmodes, n_wmodes=n_wmodes, n_tmodes=n_tmodes, embedding_dim=embedding_dim,
            ).cuda()

    else:
        raise ValueError(f'Invalid model_name {model_name}')
    
    if model_name.lower().startswith('fllronet'):
        if freeze_branchnets:
            print('Freezed BranchNets')
            net.freeze_branchnets()
        if freeze_trunknets:
            print('Freezed TrunkNets')
            net.freeze_trunknets()
        if freeze_bias:
            print('Freezed Bias')
            net.freeze_bias()

    trainer = Trainer(
        net=net, 
        lr=learning_rate,
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Configuration file name.')
    args: argparse.Namespace = parser.parse_args()
    # Load the configuration
    with open(file=args.config, mode='r') as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    # Run the main function with the configuration
    main(config)


