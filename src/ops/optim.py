from typing import Dict

import torch
from torch import optim, nn

from lion_pytorch import Lion


_Optimizer = torch.optim.Optimizer


def get_optim(optim_name: str, model: nn.Module, configs: Dict) -> _Optimizer:
    """
    optimizer 얻는 method

    :param loss_name: loss name
    :type loss_name: str
    :param model: model for obtain parameters
    :type model: nn.Module
    :param configs: config for lr
    :type configs: Dict
    :return: optimizer
    :rtype: torch.optim.Optimizer
    """
    optim_name = optim_name.lower()
    lr = configs['train']['lr']

    if optim_name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr)
    elif optim_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr)
    elif optim_name == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr)
    elif optim_name == 'lion':
        return Lion(model.parameters(), lr=lr)
    else:
        return optim.Adam(model.parameters(), lr=lr)
