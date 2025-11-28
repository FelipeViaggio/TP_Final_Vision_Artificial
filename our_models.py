import torch
import torch.nn as nn
import torch.nn.functional as F

class LarssonColorizerClassification(nn.Module):
    """
    Modelo de colorización tipo Larsson (CLASIFICACIÓN).
    
    - Input: canal L normalizado, shape (B, 1, H, W).
    - Output: logits de clases de color, shape (B, Q, H, W), Q = número de bins.
    """

    def __init__(self, backbone, num_classes):
        super().__init__()

        # Backbone preentrenado (normalmente congelado)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # Detección automática de canales de salida
        C1 = self.layer1[-1].conv3.out_channels if hasattr(self.layer1[-1], "conv3") else self.layer1[-1].conv2.out_channels
        C2 = self.layer2[-1].conv3.out_channels if hasattr(self.layer2[-1], "conv3") else self.layer2[-1].conv2.out_channels
        C3 = self.layer3[-1].conv3.out_channels if hasattr(self.layer3[-1], "conv3") else self.layer3[-1].conv2.out_channels
        C4 = self.layer4[-1].conv3.out_channels if hasattr(self.layer4[-1], "conv3") else self.layer4[-1].conv2.out_channels
        in_channels_fusion = C1 + C2 + C3 + C4

        # Fusión de features
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels_fusion, 512, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(512, 256, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
        )

        self.unsample = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(inplace=True), 
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(inplace=True), 
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(inplace=True), 
        )

        # Capa final: CLASIFICACIÓN de color
        self.output = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, L):
        """
        Args:
            L: tensor (B, 1, H, W) con el canal L normalizado.

        Returns:
            logits: tensor (B, num_classes, H, W).
        """
        B, _, H, W = L.shape
        x = L.repeat(1, 3, 1, 1)

        # Backbone
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)

        # Llevar features a igual resolución
        target_size = (8, 8)
        f1 = F.adaptive_avg_pool2d(feat1, target_size)
        f2 = F.adaptive_avg_pool2d(feat2, target_size)
        f3 = F.adaptive_avg_pool2d(feat3, target_size)
        f4 = F.adaptive_avg_pool2d(feat4, target_size)

        features = torch.cat([f1, f2, f3, f4], dim=1)

        # Fusión
        x = self.fusion(features)

        # Unsampling
        x = self.unsample(x)

        # Capa final: logits de color
        logits = self.output(x)

        # Reescalar a resolución original (H, W)
        logits = F.interpolate(
            logits, 
            size=(H, W), 
            mode="bilinear", 
            align_corners=False, 
        )

        return logits