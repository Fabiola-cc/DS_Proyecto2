"""
MODELO A: U-Net Base (Ronneberger et al. 2015)
Arquitectura del modelo
"""

import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)
    
class UNet(nn.Module):
    """U-Net original (Ronneberger et al. 2015)"""
    def __init__(self, in_channels=3, num_classes=2, base_ch=32):
            super().__init__()
            self.enc1 = DoubleConv(in_channels, base_ch)
            self.pool1 = nn.MaxPool2d(2)

            self.enc2 = DoubleConv(base_ch, base_ch*2);   self.pool2 = nn.MaxPool2d(2)
            self.enc3 = DoubleConv(base_ch*2, base_ch*4); self.pool3 = nn.MaxPool2d(2)
            self.enc4 = DoubleConv(base_ch*4, base_ch*8); self.pool4 = nn.MaxPool2d(2)

            self.bottleneck = DoubleConv(base_ch*8, base_ch*16)

            self.upconv4 = nn.ConvTranspose2d(base_ch*16, base_ch*8, 2, 2)
            self.dec4    = DoubleConv(base_ch*16, base_ch*8)
            self.upconv3 = nn.ConvTranspose2d(base_ch*8,  base_ch*4, 2, 2)
            self.dec3    = DoubleConv(base_ch*8,  base_ch*4)
            self.upconv2 = nn.ConvTranspose2d(base_ch*4,  base_ch*2, 2, 2)
            self.dec2    = DoubleConv(base_ch*4,  base_ch*2)
            self.upconv1 = nn.ConvTranspose2d(base_ch*2,  base_ch,   2, 2)
            self.dec1    = DoubleConv(base_ch*2,  base_ch)

            self.out = nn.Conv2d(base_ch, num_classes, 1)
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # Decoder con skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        return self.out(dec1)