from torch import optim
from lion_pytorch import Lion


def get_sgd(model, configs):
    return optim.SGD(model.parameters(), lr=configs['train']['lr'])


def get_adam(model, configs):
    return optim.Adam(model.parameters(), lr=configs['train']['lr'])


def get_adamw(model, configs):
    return optim.AdamW(model.parameters(), lr=configs['train']['lr'])


def get_lion(model, configs):
    return Lion(model.parameters(), lr=configs['train']['lr'])
