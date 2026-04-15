import torch
import torch.nn as nn

class MLPv01(nn.Module):
    def __init__(self, input_dim: int = 32 * 32 * 3, n_classes: int = 10):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512, bias=True),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512, bias=True),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512, bias=True),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, n_classes, bias=True),
        )
        
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, _input):
        logit = self.classifier(_input)
        return logit

class CNNv01(nn.Module):
    def __init__(self, input_dim: int = 32 * 32 * 3, n_classes: int = 10):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ), # [B, C, H, W] -> [B, OC1=32, H, W]
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0), # -> [B, OC1=32, H/2, W/2] : 16 x 16
            
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ), # [B, OC1=32, H/2, W/2] -> [B, OC2=64, H/2, W/2]
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0), # -> [B, OC2=64, H/4, W/4] : 8 x 8
            
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ), # [B, OC2=64, H/4, W/4] -> [B, OC3=128, H/4, W/4]
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0), # -> [B, OC3=128, H/8, W/8] : 4 x 4
            
            nn.Flatten(start_dim=1),
            
            nn.Linear(in_features= 128 * 4 * 4, out_features=128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(in_features= 128, out_features=128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, n_classes)
        )
        
        for layer in self.classifier:
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
    def forward(self, _input): # shape [batch, color, h, w]
        logit = self.classifier(_input)
        return logit

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int, stride: int):
        super().__init__()
        
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels
        )
        
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        
        nn.init.kaiming_uniform_(self.depthwise.weight)
        nn.init.kaiming_uniform_(self.pointwise.weight)
    
    def forward(self, x):
        return self.pointwise(self.depthwise(x))
        

class DSCNNv01(nn.Module):
    def __init__(self, input_ch: int = 3, n_classes: int = 10):
        super().__init__()
        
        
        self.classifier = nn.Sequential(
            DepthwiseSeparableConv(
                in_channels=input_ch,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            
            DepthwiseSeparableConv(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            
            DepthwiseSeparableConv(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            
            nn.Flatten(start_dim=1),
        
            nn.Linear(in_features= 128 * 4 * 4, out_features=128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(in_features= 128, out_features=128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, n_classes)  
        )
        
        for layer in self.classifier:
            if isinstance(layer, (nn.Linear)):
                nn.init.kaiming_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
    def forward(self, _input): # shape [batch, color, h, w]
        logit = self.classifier(_input)
        return logit