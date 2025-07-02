import os
import random
import logging
import numpy as np
import torch
import torch.nn as nn


def seed_all(seed=2024):
    """
    Set a fixed seed for reproducibility across random, numpy, and torch modules.
    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def get_logger(logger, logger_path, verbosity=1, name=None):
    """
    Initializes and returns a logger object based on the specified parameters.

    Args:
        logger (bool): Flag to enable or disable logging. If False, returns None.
        logger_path (str): Path where the log file will be created. Directories are created if they don't exist.
        verbosity (int): Logging verbosity level (0=DEBUG, 1=INFO, 2=WARNING). Default is 1.
        name (str): Optional name for the logger. Default is None.

    Returns:
        logging.Logger: Configured logger instance if logging is enabled, or None if disabled.

    Example:
        # Enable logging and set up a logger
        logger = get_logger(
            logger=True,
            logger_path="logs/app.log",
            verbosity=1,
            name="AppLogger"
        )
        logger.info("This is an info message.")
        logger.warning("This is a warning message.")

        # Disable logging
        logger = get_logger(logger=False, logger_path="logs/app.log")
        if logger is None:
            print("Logger is disabled.")
    """
    if logger:
        # Ensure the directory for the logger path exists
        log_dir = os.path.dirname(logger_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)  # Create the directory if it doesn't exist

        # Define log level mapping
        level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}

        # Create a formatter for log messages
        formatter = logging.Formatter(
            "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
        )

        # Create and configure the logger
        logger = logging.getLogger(name)
        logger.setLevel(level_dict[verbosity])

        # File handler for writing logs to a file
        file_handler = logging.FileHandler(logger_path, mode="w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Stream handler for writing logs to the console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # Log initialization messages
        logger.info("Logger initialized")
        logger.info(f"Log file path: {logger_path}")
    else:
        logger = None
        print("Logging is disabled")

    return logger


class MergeTemporalDim(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x_seq: torch.Tensor):
        return x_seq.flatten(0, 1).contiguous()

class ExpandTemporalDim(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T
    def forward(self, x_seq: torch.Tensor):
        y_shape = [self.T, int(x_seq.shape[0]/self.T)]
        y_shape.extend(x_seq.shape[1:])
        return x_seq.view(y_shape)


class ExpandTemporalDim_dict(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, x_dict: dict):
        # 创建一个新的字典来保存转换后的输出
        y_dict = {}
        # 遍历输入字典中的每一个键值对
        for key, x_seq in x_dict.items():
            if not isinstance(x_seq, torch.Tensor):
                raise ValueError(f"Expected a torch.Tensor but got {type(x_seq)} for key '{key}'")
            # 计算新的形状，并进行验证
            y_shape = [self.T, int(x_seq.shape[0] / self.T)]
            y_shape.extend(x_seq.shape[1:])
            y_dict[key] = x_seq.view(y_shape)
        return y_dict

class MyAt(nn.Module):
    def __init__(self):
        super(MyAt, self).__init__()
    def forward(self, x, y):
        return x @ y

class MyMul(nn.Module):
    def __init__(self):
        super(MyMul, self).__init__()
    def forward(self, x, y):
        return x * y


class MyatSequential(nn.Module):
    def __init__(self, neuron1, neuron2, module):
        super().__init__()
        self.neuron1 = neuron1
        self.neuron2 = neuron2
        self.module = module
    def forward(self, *inputs):
        return self.module(self.neuron1(inputs[0]),self.neuron2(inputs[1]))

def get_modules(nowname,model):
    flag = 0
    for name, module in model._modules.items():
        if flag==0:
            print(model.__class__.__name__.lower(),end=' ')
            flag=1
        print(module.__class__.__name__.lower(),end=' ')
    if flag==1:
        print('')
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = get_modules(name,module)
    return model
