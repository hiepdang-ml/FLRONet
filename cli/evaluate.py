import argparse
from typing import List, Tuple, Dict, Any
import yaml

from model import FLRONetFNO, FLRONetUNet, FLRONetMLP, FNO3D
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
    n_dropout_sensors: int                      = int(config['evaluate']['n_dropout_sensors'])
    noise_level: float                          = float(config['evaluate']['noise_level'])
    init_fullstate_timeframes: List[int] | None  = config['evaluate']['init_fullstate_timeframes']
    from_checkpoint: str                        = str(config['evaluate']['from_checkpoint'])

    # Load the model
    checkpoint_loader = CheckpointLoader(checkpoint_path=from_checkpoint)
    net: FLRONetFNO | FLRONetUNet | FLRONetMLP | FNO3D = checkpoint_loader.load(scope=globals())

    if isinstance(net, FNO3D):
        init_fullstate_timeframes: List[int] = list(range(min(init_sensor_timeframes), max(init_sensor_timeframes) + 1))
        n_fullstate_timeframes_per_chunk: int = len(init_fullstate_timeframes)

    # Instatiate the training datasets
    if n_dropout_sensors == 0:
        implied_dropout_probabilities: List[float] = []
    else:
        implied_dropout_probabilities: List[float] = [0.] * n_dropout_sensors
        implied_dropout_probabilities[-1] = 1.

    dataset = CFDDataset(
        root='./data/test', 
        init_sensor_timeframes=init_sensor_timeframes,
        n_fullstate_timeframes_per_chunk=n_fullstate_timeframes_per_chunk,
        n_samplings_per_chunk=n_samplings_per_chunk,
        resolution=resolution,
        n_sensors=n_sensors,
        dropout_probabilities=implied_dropout_probabilities,
        noise_level=noise_level,
        sensor_generator=sensor_generator, 
        embedding_generator=embedding_generator,
        init_fullstate_timeframes=init_fullstate_timeframes,
        seed=seed,
    )
    
    # Make prediction
    print(f'Using: {from_checkpoint}')
    predictor = Predictor(net=net)
    avg_metrics: Tuple[float, float] = predictor.predict_from_dataset(dataset)
    print(avg_metrics)


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


