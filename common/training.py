import os
import sys
import pathlib
import time
import warnings
from typing import List, Optional, Dict, TextIO, Any, Tuple, NamedTuple
from collections import defaultdict
import datetime as dt
import copy
import inspect

import torch
import torch.nn as nn


class Accumulator:
    """
    A utility class for accumulating values for multiple metrics.
    """

    def __init__(self) -> None:
        self.__records: defaultdict[str, float] = defaultdict(float)

    def add(self, **kwargs: Any) -> None:
        """
        Add values to the accumulator.

        Parameters:
            - **kwargs: named metric and the value is the amount to add.
        """
        metric: str
        value: float
        for metric, value in kwargs.items():
            # Each keyword argument represents a metric name and its value to be added
            self.__records[metric] += value
    
    def reset(self) -> None:
        """
        Reset the accumulator by clearing all recorded metrics.
        """
        self.__records.clear()

    def __getitem__(self, key: str) -> float:
        """
        Retrieve a record by key.

        Parameters:
            - key (str): The record key name.

        Returns:
            - float: The record value.
        """
        return self.__records[key]


class EarlyStopping:
    """
    A simple early stopping utility to terminate training when a monitored metric stops improving.

    Attributes:
        - patience (int): The number of epochs with no improvement after which training will be stopped.
        - tolerance (float): The minimum change in the monitored metric to qualify as an improvement,
        - considering the direction of the metric being monitored.
        - bestscore (float): The best score seen so far.
    """
    
    def __init__(self, patience: int, tolerance: float = 0.) -> None:
        """
        Initializes the EarlyStopping instance.
        
        Parameters:
            - patience (int): Number of epochs with no improvement after which training will be stopped.
            - tolerance (float): The minimum change in the monitored metric to qualify as an improvement. 
            Defaults to 0.
        """
        self.patience: int = patience
        self.tolerance: float = tolerance
        self.bestscore: float = float('inf')
        self.__counter: int = 0

    def __call__(self, value: float) -> None:
        """
        Update the state of the early stopping mechanism based on the new metric value.

        Parameters:
            - value (float): The latest value of the monitored metric.
        """
        # Improvement or within tolerance, reset counter
        if value <= self.bestscore + self.tolerance:
            self.bestscore: float = value
            self.__counter: int = 0

        # No improvement, increment counter
        else:
            self.__counter += 1

    def __bool__(self) -> bool:
        """
        Determine if the training process should be stopped early.

        Returns:
            - bool: True if training should be stopped (patience exceeded), otherwise False.
        """
        return self.__counter >= self.patience


class Timer:

    """
    A class used to time the duration of epochs and batches.
    """
    def __init__(self) -> None:
        """
        Initialize the Timer.
        """
        self.__epoch_starts: Dict[int, float] = dict()
        self.__epoch_ends: Dict[int, float] = dict()
        self.__batch_starts: Dict[int, Dict[int, float]] = defaultdict(dict)
        self.__batch_ends: Dict[int, Dict[int, float]] = defaultdict(dict)

    def start_epoch(self, epoch: int) -> None:
        """
        Start timing an epoch.

        Parameters:
            epoch (int): The epoch number.
        """
        self.__epoch_starts[epoch] = time.time()

    def end_epoch(self, epoch: int) -> None:
        """
        End timing an epoch.

        Parameters:
            - epoch (int): The epoch number.
        """
        self.__epoch_ends[epoch] = time.time()

    def start_batch(self, epoch: int, batch: Optional[int] = None) -> None:
        """
        Start timing a batch.

        Parameters:
            - epoch (int): The epoch number.
            - batch (int, optional): The batch number. If not provided, the next batch number is used.
        """
        if batch is None:
            if self.__batch_starts[epoch]:
                batch: int = max(self.__batch_starts[epoch].keys()) + 1
            else:
                batch: int = 1
        self.__batch_starts[epoch][batch] = time.time()
    
    def end_batch(self, epoch: int, batch: Optional[int] = None) -> None:
        """
        End timing a batch.

        Parameters:
            - epoch (int): The epoch number.
            - batch (int, optional): The batch number. If not provided, the last started batch number is used.
        """
        if batch is None:
            if self.__batch_starts[epoch]:
                batch: int = max(self.__batch_starts[epoch].keys())
            else:
                raise RuntimeError(f"no batch has started")
        self.__batch_ends[epoch][batch] = time.time()
    
    def time_epoch(self, epoch: int) -> float:
        """
        Get the duration of an epoch.

        Parameters:
            - epoch (int): The epoch number.

        Returns:
            - float: The duration of the epoch in seconds.
        """
        result: float = self.__epoch_ends[epoch] - self.__epoch_starts[epoch]
        if result > 0:
            return result
        else:
            raise RuntimeError(f"epoch {epoch} ends before starts")
    
    def time_batch(self, epoch: int, batch: int) -> float:
        """
        Get the duration of a batch.

        Parameters:
            - epoch (int): The epoch number.
            - batch (int): The batch number.

        Returns:
            - float: The duration of the batch in seconds.
        """
        result: float = self.__batch_ends[epoch][batch] - self.__batch_starts[epoch][batch]
        if result > 0:
            return result
        else:
            raise RuntimeError(f"batch {batch} in epoch {epoch} ends before starts")
        

class Logger:

    """
    A class used to log the training process.

    This class provides methods to log messages to a file and the console. 
    """
    def __init__(self, logfile: str = f".logs/{dt.datetime.now().strftime('%Y%m%d%H%M%S')}") -> None:
    
        """
        Initialize the logger.

        Parameters:
            - logfile (str, optional): The path to the logfile. 
            Defaults to a file in the .logs directory with the current timestamp.
        """
        self.logfile: pathlib.Path = pathlib.Path(logfile)
        os.makedirs(name=self.logfile.parent, exist_ok=True)
        self._file: TextIO = open(self.logfile, mode='w')

    def log(
        self, 
        epoch: int, 
        n_epochs: int, 
        batch: Optional[int] = None, 
        n_batches: Optional[int] = None, 
        took: Optional[float] = None, 
        **kwargs: Any,
    ) -> None:
        """
        Log a message to console and a log file

        Parameters:
            - epoch (int): The current epoch.
            - n_epochs (int): The total number of epochs.
            - batch (int, optional): The current batch. Defaults to None.
            - n_batches (int, optional): The total number of batches. Defaults to None.
            - took (float, optional): The time it took to process the batch or epoch. Defaults to None.
            - **kwargs: Additional metrics to log.
        """
        suffix: str = ', '.join([f'{metric}: {value:.3e}' for metric, value in kwargs.items()])
        prefix: str = f'Epoch {epoch}/{n_epochs} | '
        if batch is not None:
            prefix += f'Batch {batch}/{n_batches} | '
        if took is not None:
            prefix += f'Took {took:.2f}s | '
        logstring: str = prefix + suffix
        self._file.write(logstring + '\n')

    def __del__(self) -> None:
        """
        Close the logfile at garbage collected.
        """
        self._file.close()


class CheckpointSaver:
    """
    A class used to save PyTorch model checkpoints.
    """
    def __init__(self, model: nn.Module, dirpath: str) -> None:
        """
        Initialize the CheckPointSaver.

        Parameters:
            - model (nn.Module): The class object of the model
            - dirpath (os.PathLike): The directory where the checkpoints to save.
        """
        self.dirpath: pathlib.Path = pathlib.Path(dirpath)
        # For model reconstruction
        self.model_classname: str = model.__class__.__name__
        signature: inspect.Signature = inspect.signature(model.__init__)
        self.model_kwargs: Dict[str, Any] = {
            p: getattr(model, p) for p in signature.parameters.keys() if p != 'self'
        }
        # ensure the dirpath exists in the file system
        os.makedirs(name=self.dirpath, exist_ok=True)

    def save(self, model_states: Dict[str, Any], filename: str) -> None:
        """
        Save checkpoint to a .pt file.

        Parameters:
            - model_states (Dict[str, torch.Tensor]): The output of model.state_dict()
            - filename (str): the checkpoint file name
        """
        torch.save(
            obj={
                'model': {
                    'classname' : self.model_classname,
                    'kwargs'    : self.model_kwargs,
                    'states'    : copy.deepcopy(model_states),
                },
            },
            f=os.path.join(self.dirpath, filename)
        )


class CheckpointLoader:
    """
    A class used to load PyTorch model checkpoints.
    """
    def __init__(self, checkpoint_path: str) -> None:
        """
        Initialize the CheckpointLoader.

        Parameters:
            - checkpoint_path (str): The path to the checkpoint file.
        """
        self.checkpoint_path: str = checkpoint_path
        self.__checkpoint: Dict[str, Any] = torch.load(checkpoint_path, weights_only=False)

        # Model metadata
        self.model_classname: str = self.__checkpoint['model']['classname']
        self.model_kwargs: Dict[str, Any] = self.__checkpoint['model']['kwargs']

    def load(
        self, 
        scope: Dict[str, Any], 
        ignored_modules: List[str] = [], 
        **overrided_params: Dict[str, Any]
    ) -> nn.Module:
        """
        Load the model from the checkpoint.

        Parameters:
            - scope (Dict[str, Any]): The namespace to look up the model object. 
                It's typically the dictionary output of `globals()` or `locals()`
            - ignored_modules (List[str]): names of modules that are excluded from loading.
                Default is []
            - **overrided_params (Dict[str, Any]): overrid loaded model parameters (for later retrain). 
                Default is {}, which means keeping the original model parameters unchanged
        
        Returns:
            - nn.Module: The model loaded from the checkpoint.
        """
        # Check caller's namespace for model object
        if self.model_classname not in scope.keys():
            raise ImportError(
                f'{self.model_classname} is not found in the current namespace, you might need to import it first.'
            )
        
        # Instantiate model
        if overrided_params:
            print({'Original': self.model_kwargs, 'Changed params': overrided_params})
            if input('Enter "yes" to confirm new model parameters: ').lower() == 'yes':
                self.model_kwargs.update(overrided_params)
            else:
                sys.exit()
        print(self.model_kwargs)
        model: nn.Module = eval(self.model_classname, scope)(**self.model_kwargs)

        # Load model from model state_dict and check for compatibility
        model_states: Dict[str, Any] = self.__checkpoint['model']['states']
        for ignored_module in ignored_modules:
            model_states = {k: v for k, v in model_states.items() if not k.startswith(ignored_module)}

        model_incompatible_keys: NamedTuple = model.load_state_dict(model_states, strict=False)   # inplace update
        if model_incompatible_keys.missing_keys:  # List[str]
            warnings.warn(
                f'Missing keys from the loaded model checkpoint: {model_incompatible_keys.missing_keys}', 
                category=UserWarning
            )
        if model_incompatible_keys.unexpected_keys: # List[str]
            warnings.warn(
                f'Unexpected keys found in the loaded model checkpoint: {model_incompatible_keys.unexpected_keys}', 
                category=UserWarning
            )
        
        return model