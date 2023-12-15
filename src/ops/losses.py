from torch import nn


def get_multi_loss():
    return nn.MultiLabelSoftMarginLoss()


def get_loss():
    return nn.CrossEntropyLoss()


def get_bce_loss():
    return nn.BCELoss()
