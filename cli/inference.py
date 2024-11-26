import argparse
from typing import Tuple, List, Dict, Any
import yaml

from model import FLRONetFNO, FLRONetUNet, FLRONetMLP, FNO3D
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
    sensor_position_path: str                   = str(config['inference']['sensor_position_path'])
    n_dropout_sensors: int                      = int(config['inference']['n_dropout_sensors'])
    noise_level: float                          = float(config['inference']['noise_level'])
    from_checkpoint: str                        = str(config['inference']['from_checkpoint'])
    trained_resolution: Tuple[int, int]         = tuple(config['dataset']['resolution'])
    out_resolution: Tuple[int, int]             = tuple(config['inference']['out_resolution'])

    # Load the model
    print(f'Using: {from_checkpoint}')
    checkpoint_loader = CheckpointLoader(checkpoint_path=from_checkpoint)
    net: FLRONetFNO | FLRONetUNet | FLRONetMLP | FNO3D = checkpoint_loader.load(scope=globals())
    
    # Make prediction
    predictor = Predictor(net=net)
    predictor.predict_from_scratch(
        case_dir=case_dir,
        sensor_timeframes=sensor_timeframes,
        reconstruction_timeframes=reconstruction_timeframes,
        sensor_position_path=sensor_position_path,
        embedding_generator=embedding_generator,
        n_dropout_sensors=n_dropout_sensors,
        noise_level=noise_level,
        in_resolution=trained_resolution,
        out_resolution=out_resolution if isinstance(net, FLRONetFNO) else None,
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

