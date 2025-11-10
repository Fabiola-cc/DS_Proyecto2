# modeloC_tiny.py
"""
Attention U-Net Mínimo - Para GPUs de 2GB o menos
Aproximadamente 2M parámetros
"""

import torch
import torch.nn as nn


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1)
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1)
        self.psi = nn.Conv2d(F_int, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.sigmoid(self.psi(psi))
        return x * psi


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        return x


class AttentionUNetTiny(nn.Module):
    """
    Versión ultra-pequeña
    Canales: [16, 32, 64, 128] - Cuatro veces más pequeño
    Solo 3 niveles en lugar de 4
    """
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()
        
        # Encoder - Solo 3 niveles
        self.enc1 = ConvBlock(in_channels, 16)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = ConvBlock(16, 32)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.enc3 = ConvBlock(32, 64)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = ConvBlock(64, 128)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.att3 = AttentionGate(64, 64, 32)
        self.dec3 = ConvBlock(128, 64)
        
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.att2 = AttentionGate(32, 32, 16)
        self.dec2 = ConvBlock(64, 32)
        
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.att1 = AttentionGate(16, 16, 8)
        self.dec1 = ConvBlock(32, 16)
        
        self.out = nn.Conv2d(16, num_classes, 1)
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool3(enc3))
        
        # Decoder
        dec3 = self.up3(bottleneck)
        enc3_att = self.att3(g=dec3, x=enc3)
        dec3 = torch.cat([enc3_att, dec3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.up2(dec3)
        enc2_att = self.att2(g=dec2, x=enc2)
        dec2 = torch.cat([enc2_att, dec2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.up1(dec2)
        enc1_att = self.att1(g=dec1, x=enc1)
        dec1 = torch.cat([enc1_att, dec1], dim=1)
        dec1 = self.dec1(dec1)
        
        return self.out(dec1)


if __name__ == "__main__":
    model = AttentionUNetTiny()
    total = sum(p.numel() for p in model.parameters())
    print(f"Parámetros: {total:,} ({total/1e6:.2f}M)")
    
    x = torch.randn(1, 3, 512, 512)
    y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")