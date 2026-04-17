import torch
import torch.nn as nn
import torchvision.models as ftmodels
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool


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


def build_mobilenet_fpn_backbone():
    backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)

    # MobileNetV3's feature layers are inside backbone.features
    # We extract intermediate layers by name for the FPN
    # These layer indices correspond to strides 8, 16, and 32
    return_layers = {
        '3':  '0',   # stride 8  — fine details, good for Platelets
        '7':  '1',   # stride 16 — medium features, good for RBCs
        '13': '2',   # stride 32 — coarse semantics, good for WBCs
    }

    # Channel sizes at each extracted layer in MobileNetV3-Large
    in_channels_list = [24, 48, 96]
    out_channels     = 256  # FPN normalizes all levels to this channel count

    backbone_with_fpn = BackboneWithFPN(
        backbone=backbone.features,
        return_layers=return_layers,
        in_channels_list=in_channels_list,
        out_channels=out_channels,
        extra_blocks=LastLevelMaxPool()  # adds one more pooled level for very large objects
    )

    return backbone_with_fpn


def build_anchor_generator():    
    # Platelets are tiny: we need small anchors (16, 32 px)
    # RBCs are medium: 32, 64 px
    # WBCs are large: 64, 128, 256 px
    # Aspect ratios: all three cell types are roughly circular, so 0.5, 1.0, 2.0
    anchor_generator = AnchorGenerator(
        sizes=((16, 32), (32, 64), (64, 128), (128, 256), (256, 512)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5  # same ratios at every level
    )
    return anchor_generator


def build_model(num_classes):

    backbone         = build_mobilenet_fpn_backbone()
    anchor_generator = build_anchor_generator()

    # RoI Align output size — each proposed region gets cropped to 7x7
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],  # which FPN levels to pool from
        output_size=7,
        sampling_ratio=2
    )

    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,

        # RPN hyperparameters — these control how proposals are filtered
        rpn_pre_nms_top_n_train=2000,   # keep top 2000 proposals before NMS during training
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=1000,  # keep top 1000 after NMS during training
        rpn_post_nms_top_n_test=300,
        rpn_nms_thresh=0.7,             # IoU threshold for RPN NMS
        rpn_fg_iou_thresh=0.7,          # anchor is "foreground" if IoU with GT > 0.7
        rpn_bg_iou_thresh=0.3,          # anchor is "background" if IoU with GT < 0.3
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,

        # Detection head hyperparameters
        box_score_thresh=0.05,     # low threshold during training to see all predictions
        box_nms_thresh=0.5,        # IoU threshold for final NMS
        box_detections_per_img=100 # max 100 detections per image
    )

    return model


def get_model_objectdetection_mobilenet(num_classes=4, device='cpu'):
    model = build_model(num_classes)
    model = model.to(device)

    # Count parameters as a sanity check
    total  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")

    return model