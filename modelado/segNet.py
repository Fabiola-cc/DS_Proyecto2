"""
MODELO B: SegNet (Badrinarayanan et al. 2017)
Arquitectura del modelo con max pooling indices
"""

import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
    """Bloque encoder de SegNet con BatchNorm"""
    def __init__(self, in_channels, out_channels, num_convs=2):
        super().__init__()
        layers = []
        for i in range(num_convs):
            layers.extend([
                nn.Conv2d(in_channels if i == 0 else out_channels, 
                         out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
        self.conv_block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.conv_block(x)

class DecoderBlock(nn.Module):
    """Bloque decoder de SegNet con BatchNorm"""
    def __init__(self, in_channels, out_channels, num_convs=2):
        super().__init__()
        layers = []
        for i in range(num_convs):
            layers.extend([
                nn.Conv2d(in_channels if i == 0 else out_channels,
                         out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
        self.conv_block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.conv_block(x)

class SegNet(nn.Module):
    """
    SegNet (Badrinarayanan et al. 2017)
    Usa índices de max pooling para upsampling en lugar de skip connections
    """
    def __init__(self, in_channels=3, num_classes=2):
        super(SegNet, self).__init__()
        
        # ENCODER
        self.enc1 = EncoderBlock(in_channels, 64, num_convs=2)
        self.pool1 = nn.MaxPool2d(2, stride=2, return_indices=True)
        
        self.enc2 = EncoderBlock(64, 128, num_convs=2)
        self.pool2 = nn.MaxPool2d(2, stride=2, return_indices=True)
        
        self.enc3 = EncoderBlock(128, 256, num_convs=3)
        self.pool3 = nn.MaxPool2d(2, stride=2, return_indices=True)
        
        self.enc4 = EncoderBlock(256, 512, num_convs=3)
        self.pool4 = nn.MaxPool2d(2, stride=2, return_indices=True)
        
        self.enc5 = EncoderBlock(512, 512, num_convs=3)
        self.pool5 = nn.MaxPool2d(2, stride=2, return_indices=True)
        
        # DECODER (usa los índices del encoder para upsampling)
        self.unpool5 = nn.MaxUnpool2d(2, stride=2)
        self.dec5 = DecoderBlock(512, 512, num_convs=3)
        
        self.unpool4 = nn.MaxUnpool2d(2, stride=2)
        self.dec4 = DecoderBlock(512, 256, num_convs=3)
        
        self.unpool3 = nn.MaxUnpool2d(2, stride=2)
        self.dec3 = DecoderBlock(256, 128, num_convs=3)
        
        self.unpool2 = nn.MaxUnpool2d(2, stride=2)
        self.dec2 = DecoderBlock(128, 64, num_convs=2)
        
        self.unpool1 = nn.MaxUnpool2d(2, stride=2)
        self.dec1 = DecoderBlock(64, 64, num_convs=2)
        
        # Output layer
        self.out = nn.Conv2d(64, num_classes, kernel_size=1)
    
    def forward(self, x):
        # ENCODER con almacenamiento de índices
        enc1 = self.enc1(x)
        enc1_pooled, indices1 = self.pool1(enc1)
        
        enc2 = self.enc2(enc1_pooled)
        enc2_pooled, indices2 = self.pool2(enc2)
        
        enc3 = self.enc3(enc2_pooled)
        enc3_pooled, indices3 = self.pool3(enc3)
        
        enc4 = self.enc4(enc3_pooled)
        enc4_pooled, indices4 = self.pool4(enc4)
        
        enc5 = self.enc5(enc4_pooled)
        enc5_pooled, indices5 = self.pool5(enc5)
        
        # DECODER usando índices almacenados
        dec5 = self.unpool5(enc5_pooled, indices5)
        dec5 = self.dec5(dec5)
        
        dec4 = self.unpool4(dec5, indices4)
        dec4 = self.dec4(dec4)
        
        dec3 = self.unpool3(dec4, indices3)
        dec3 = self.dec3(dec3)
        
        dec2 = self.unpool2(dec3, indices2)
        dec2 = self.dec2(dec2)
        
        dec1 = self.unpool1(dec2, indices1)
        dec1 = self.dec1(dec1)
        
        return self.out(dec1)


# Ejemplo de uso
if __name__ == "__main__":
    # Crear modelo
    model = SegNet(in_channels=3, num_classes=2)
    
    # Crear tensor de entrada de ejemplo (batch_size, channels, height, width)
    x = torch.randn(1, 3, 256, 256)
    
    # Forward pass
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")