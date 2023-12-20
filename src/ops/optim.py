from torch import optim
from lion_pytorch import Lion


def get_optim(optim_name, model, configs):
    optim_name = optim_name.lower()

    if optim_name == 'sgd':
        return optim.SGD(model.parameters(), lr=configs['train']['lr'])
    elif optim_name == 'adam':
        return optim.Adam(model.parameters(), lr=configs['train']['lr'])
    elif optim_name == 'adamw':
        return optim.AdamW(model.parameters(), lr=configs['train']['lr'])
    elif optim_name == 'lion':
        return Lion(model.parameters(), lr=configs['train']['lr'])
    else:
        return optim.Adam(model.parameters(), lr=configs['train']['lr'])
