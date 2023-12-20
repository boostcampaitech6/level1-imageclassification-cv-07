import timm
from torch import nn


class MultiLabelModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            'tf_efficientnet_b4_ns', pretrained=True
        )
        num_in_features = self.model.get_classifier().out_features
        self.fc_1 = nn.Sequential(
            nn.Linear(num_in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 3)
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(num_in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 2)
        )
        self.fc_3 = nn.Sequential(
            nn.Linear(num_in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 3)
        )
        self.name = 'tf_efficientnet_b4_ns_mask_multilabel'

    def forward(self, x):
        x = self.model(x)
        output1 = self.fc_1(x)
        output2 = self.fc_2(x)
        output3 = self.fc_3(x)
        return output1, output2, output3


class SingleLabelModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            'convnext_small.fb_in22k', pretrained=True, num_classes=18
        )
        self.name = 'convnext_small.fb_in22k_singlelabel'

    def forward(self, x):
        x = self.model(x)
        return x


class ResNet50Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            'resnet50.a1_in1k', pretrained=True, num_classes=18
        )
        self.name = 'resnet50.a1_in1k_singlelabel'

    def forward(self, x):
        x = self.model(x)
        return x


class EfficientNetB4Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            'tf_efficientnet_b4.ns_jft_in1k', pretrained=True, num_classes=18
        )
        self.name = 'tf_efficientnet_b4.ns_jft_in1k_singlelabel'

    def forward(self, x):
        x = self.model(x)
        return x


class EfficientNetB7Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            'tf_efficientnet_b7.ns_jft_in1k', pretrained=True, num_classes=18
        )
        self.name = 'tf_efficientnet_b7.ns_jft_in1k_singlelabel'

    def forward(self, x):
        x = self.model(x)
        return x


class VITModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            'vit_base_patch16_224.augreg2_in21k_ft_in1k',
            pretrained=True, num_classes=18
        )
        self.name = 'vit_base_patch16_224.augreg2_in21k_ft_in1k_singlelabel'

    def forward(self, x):
        x = self.model(x)
        return x


class SwinTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            'swin_base_patch4_window7_224.ms_in22k_ft_in1k',
            pretrained=True, num_classes=18
        )
        self.name = 'swin_base_patch4_window7_224.ms_in22k_ft_in1k_singlelabel'

    def forward(self, x):
        x = self.model(x)
        return x
