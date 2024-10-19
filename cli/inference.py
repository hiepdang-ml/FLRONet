import argparse
from typing import Tuple, List, Dict, Any

import yaml
from torch.optim import Optimizer, Adam

from cfd.embedding import Mask, Voronoi
from model.flronet import FLRONetWithFNO, FLRONetWithUNet
from common.training import CheckpointLoader
from worker import Predictor


def main(config: Dict[str, Any]) -> None:
    """
    Main function to reconstruct the fullstate at any frames from sensor measurements.

    Parameters:
        config (Dict[str, Any]): Configuration dictionary.
    """

    # Parse CLI arguments:
    embedding_generator: str                    = str(config['dataset']['embedding_generator'])
    case_dir: str                               = str(config['inference']['case_dir'])
    sensor_timeframes: List[int]                = list(config['inference']['sensor_timeframes'])
    reconstruction_timeframes: List[int]        = list(config['inference']['reconstruction_timeframes'])
    resolution: Tuple[int, int]                 = tuple(config['inference']['resolution'])
    sensor_position_path: str                   = str(config['inference']['sensor_position_path'])
    from_checkpoint: str                        = str(config['inference']['from_checkpoint'])

    # Instatiate the embedding generator
    if embedding_generator == 'Mask':
        embedding_generator = Mask()
    elif embedding_generator == 'Voronoi':
        embedding_generator = Voronoi()
    else:
        raise ValueError(f'Invalid embedding_generator: {embedding_generator}')

    # Load the model
    checkpoint_loader = CheckpointLoader(checkpoint_path=from_checkpoint)
    net: FLRONetWithFNO | FLRONetWithUNet = checkpoint_loader.load(scope=globals())[0]
    
    # Make prediction
    predictor = Predictor(
        net=net,
        sensor_position_path=sensor_position_path,
        embedding_generator=embedding_generator
    )
    predictor.predict_from_scratch(
        case_dir=case_dir,
        sensor_timeframes=sensor_timeframes,
        reconstruction_timeframes=reconstruction_timeframes,
        resolution=resolution
    )


if __name__ == "__main__":
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Reconstruct the fullstate at any frames from sensor measurements')
    parser.add_argument('--config', type=str, required=True, help='Configuration file name.')
    args: argparse.Namespace = parser.parse_args()
    # Load the configuration
    with open(file=args.config, mode='r') as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    # Run the main function with the configuration
    main(config)

