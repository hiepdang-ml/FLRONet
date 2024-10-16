import argparse
from typing import List, Dict, Any

import yaml
from torch.optim import Optimizer, Adam

from cfd.sensors import LHS, AroundCylinder
from cfd.embedding import Mask, Voronoi
from model.flronet import FLRONet
from cfd.dataset import CFDDataset
from common.training import CheckpointLoader
from worker import Predictor


def main(config: Dict[str, Any]) -> None:
    """
    Main function to evaluate a trained FLRONet on test dataset.

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
    seed: int                                   = int(config['dataset']['seed'])
    already_preloaded: bool                     = bool(config['dataset']['already_preloaded'])

    from_checkpoint: str                        = str(config['test']['from_checkpoint'])

    # Instatiate the sensor generator
    if sensor_generator == 'AroundCylinder':
        sensor_generator = AroundCylinder(n_sensors)
    elif sensor_generator == 'LHS':
        sensor_generator = LHS(n_sensors)
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
    dataset = CFDDataset(
        root='./data/test', 
        init_sensor_timeframes=init_sensor_timeframes,
        n_fullstate_timeframes_per_chunk=n_fullstate_timeframes_per_chunk,
        n_samplings_per_chunk=n_samplings_per_chunk,
        resolution=resolution,
        sensor_generator=sensor_generator, 
        embedding_generator=embedding_generator,
        seed=seed,
        already_preloaded=already_preloaded,
    )

    # Load the model
    checkpoint_loader = CheckpointLoader(checkpoint_path=from_checkpoint)
    net: FLRONet = checkpoint_loader.load(scope=globals())[0]
        
    # Make prediction
    predictor = Predictor(net=net)
    predictor.predict_from_dataset(dataset)


if __name__ == "__main__":
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Evaluate a trained FLRONet on test dataset')
    parser.add_argument('--config', type=str, required=True, help='Configuration file name.')
    args: argparse.Namespace = parser.parse_args()
    # Load the configuration
    with open(file=args.config, mode='r') as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    # Run the main function with the configuration
    main(config)


