import timm
from torch import nn


class MaskModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            'tf_efficientnet_b4_ns', pretrained=True
        )
        num_in_features = self.model.get_classifier().out_features
        self.fc = nn.Linear(
            in_features=num_in_features, out_features=8, bias=False
        )
        self.fc_1 = nn.Linear(8, 3)
        self.fc_2 = nn.Linear(8, 2)
        self.fc_3 = nn.Linear(8, 3)
        self.name = 'tf_efficientnet_b4_ns_maskv1'

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        output1 = self.fc_1(x)
        output2 = self.fc_2(x)
        output3 = self.fc_3(x)
        return output1, output2, output3


class MaskModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            'tf_efficientnet_b4_ns', pretrained=True
        )
        num_in_features = self.model.get_classifier().out_features
        self.fc_1 = nn.Linear(num_in_features, 3)
        self.fc_2 = nn.Linear(num_in_features, 2)
        self.fc_3 = nn.Linear(num_in_features, 3)
        self.name = 'tf_efficientnet_b4_ns_maskv2'

    def forward(self, x):
        x = self.model(x)
        output1 = self.fc_1(x)
        output2 = self.fc_2(x)
        output3 = self.fc_3(x)
        return output1, output2, output3


class MaskModelV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            'tf_efficientnet_b4_ns', pretrained=True
        )
        num_in_features = self.model.get_classifier().out_features
        self.fc = nn.Sequential(
            nn.Linear(num_in_features, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 128)

        )
        self.fc_1 = nn.Linear(128, 3)
        self.fc_2 = nn.Linear(128, 2)
        self.fc_3 = nn.Linear(128, 3)
        self.name = 'tf_efficientnet_b4_ns_maskv3'

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        output1 = self.fc_1(x)
        output2 = self.fc_2(x)
        output3 = self.fc_3(x)
        return output1, output2, output3


class MaskModelV4(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            'tf_efficientnet_b4_ns', pretrained=True
        )
        num_in_features = self.model.get_classifier().out_features
        self.fc = nn.Sequential(
            nn.Linear(num_in_features, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 18),
        )
        self.name = 'tf_efficientnet_b4_ns_maskv4'

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x


class MaskModelV5(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            'tf_efficientnet_b4_ns', pretrained=True
        )
        num_in_features = self.model.get_classifier().out_features
        self.fc = nn.Linear(num_in_features, 8)
        self.name = 'tf_efficientnet_b4_ns_maskv4'

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x
