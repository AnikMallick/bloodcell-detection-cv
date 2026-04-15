import torch
import torch.nn as nn
import torchvision.models as ftmodels


class ResNetFTv01(nn.Module):
    def __init__(self, n_classes: int = 10):
        super().__init__()
        
        base_model = ftmodels.resnet18(weights="DEFAULT")
        
        base_model.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        base_model.maxpool = nn.Identity()
        
        out_feature = base_model.fc.in_features
        base_model.fc = nn.Identity()
        
        self.head = nn.Sequential(
            nn.Linear(out_feature, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            
            nn.Linear(128, n_classes),
        )
        
        self.classifier = nn.Sequential(
            base_model,
            self.head
        )
        
        for param in base_model.parameters():
            param.requires_grad = False
        
        for param in base_model.conv1.parameters():
            param.requires_grad = True
        
        
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)
                if layer.bias != None:
                    nn.init.zeros_(layer.bias)
        
        nn.init.kaiming_uniform_(base_model.conv1.weight)
        
    def forward(self, _input): # shape [batch, color, h, w]
        logit = self.classifier(_input)
        return logit
        
class ResNetFTv02(nn.Module):
    def __init__(self, n_classes: int = 10):
        super().__init__()
        
        base_model = ftmodels.resnet18(weights="DEFAULT")
        
        
        out_feature = base_model.fc.in_features
        base_model.fc = nn.Identity()
        
        self.head = nn.Sequential(
            nn.Linear(out_feature, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            
            nn.Linear(128, n_classes),
        )
        
        self.classifier = nn.Sequential(
            base_model,
            self.head
        )
        
        for param in base_model.parameters():
            param.requires_grad = False       
        
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)
                if layer.bias != None:
                    nn.init.zeros_(layer.bias)
        
    def forward(self, _input): # shape [batch, color, h, w]
        logit = self.classifier(_input)
        return logit


        
class ResNetFTv03(nn.Module):
    def __init__(self, n_classes: int = 10):
        super().__init__()
        
        self.base_model = ftmodels.resnet18(weights="DEFAULT")
        
        self.base_model.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.base_model.maxpool = nn.Identity()
        
        out_feature = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()
        
        self.head = nn.Sequential(
            nn.Linear(out_feature, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            
            nn.Linear(128, n_classes),
        )
        
        self.classifier = nn.Sequential(
            self.base_model,
            self.head
        )
        
        for param in self.base_model.parameters():
            param.requires_grad = False       
        
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)
                if layer.bias != None:
                    nn.init.zeros_(layer.bias)
        
    def forward(self, _input): # shape [batch, color, h, w]
        logit = self.classifier(_input)
        return logit