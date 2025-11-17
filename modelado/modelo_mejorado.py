"""
MODELO MEJORADO: AttentionResUNet
Combina lo mejor de U-Net, SegNet y añade mejoras modernas:
- Attention Gates para enfocarse en regiones relevantes
- Residual Connections para mejor flujo de gradientes
- Deep Supervision para mejor entrenamiento
- GroupNorm en lugar de BatchNorm
- Arquitectura más profunda y robusta
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Bloque residual con GroupNorm y ReLU"""
    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(min(groups, out_channels), out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(min(groups, out_channels), out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.GroupNorm(min(groups, out_channels), out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.gn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out


class AttentionGate(nn.Module):
    """
    Attention Gate para enfocarse en regiones relevantes
    Ref: Attention U-Net (Oktay et al. 2018)
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(min(8, F_int), F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(min(8, F_int), F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(1, 1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        """
        g: gating signal (del decoder)
        x: skip connection (del encoder)
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class EncoderBlock(nn.Module):
    """Bloque encoder con residual connections"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.res_block1 = ResidualBlock(in_channels, out_channels)
        self.res_block2 = ResidualBlock(out_channels, out_channels)
        self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)
    
    def forward(self, x):
        x = self.res_block1(x)
        x = self.res_block2(x)
        x_pooled, indices = self.pool(x)
        return x, x_pooled, indices


class DecoderBlock(nn.Module):
    """Bloque decoder con attention gates y residual connections"""
    def __init__(self, in_channels, out_channels, skip_channels):
        super().__init__()
        
        # Upsampling
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
        # Attention gate
        self.attention = AttentionGate(F_g=out_channels, F_l=skip_channels, F_int=out_channels//2)
        
        # Residual blocks
        self.res_block1 = ResidualBlock(out_channels + skip_channels, out_channels)
        self.res_block2 = ResidualBlock(out_channels, out_channels)
    
    def forward(self, x, skip):
        x = self.up(x)
        
        # Aplicar attention gate a skip connection
        skip = self.attention(g=x, x=skip)
        
        # Concatenar con skip connection
        x = torch.cat([x, skip], dim=1)
        
        x = self.res_block1(x)
        x = self.res_block2(x)
        
        return x


class AttentionResUNet(nn.Module):
    """
    Modelo mejorado para segmentación de vasos sanguíneos
    
    Características:
    - Residual blocks para mejor entrenamiento
    - Attention gates para enfocarse en vasos
    - Deep supervision para múltiples escalas
    - GroupNorm para mejor normalización
    """
    def __init__(self, in_channels=3, num_classes=2, base_ch=64):
        super().__init__()
        
        # Encoder path
        self.enc1 = EncoderBlock(in_channels, base_ch)
        self.enc2 = EncoderBlock(base_ch, base_ch*2)
        self.enc3 = EncoderBlock(base_ch*2, base_ch*4)
        self.enc4 = EncoderBlock(base_ch*4, base_ch*8)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(base_ch*8, base_ch*16),
            ResidualBlock(base_ch*16, base_ch*16)
        )
        
        # Decoder path con attention
        self.dec4 = DecoderBlock(base_ch*16, base_ch*8, base_ch*8)
        self.dec3 = DecoderBlock(base_ch*8, base_ch*4, base_ch*4)
        self.dec2 = DecoderBlock(base_ch*4, base_ch*2, base_ch*2)
        self.dec1 = DecoderBlock(base_ch*2, base_ch, base_ch)
        
        # Deep supervision outputs (para entrenar mejor)
        self.out_final = nn.Conv2d(base_ch, num_classes, kernel_size=1)
        self.out_dec2 = nn.Conv2d(base_ch*2, num_classes, kernel_size=1)
        self.out_dec3 = nn.Conv2d(base_ch*4, num_classes, kernel_size=1)
    
    def forward(self, x, return_deep_supervision=False):
        """
        Args:
            x: Input image [B, 3, H, W]
            return_deep_supervision: Si True, retorna outputs auxiliares
        
        Returns:
            output: Predicción final [B, num_classes, H, W]
            (opcional) outputs auxiliares para deep supervision
        """
        input_size = x.size()
        
        # Encoder
        skip1, x, idx1 = self.enc1(x)
        skip2, x, idx2 = self.enc2(x)
        skip3, x, idx3 = self.enc3(x)
        skip4, x, idx4 = self.enc4(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder con attention
        x = self.dec4(x, skip4)
        
        x = self.dec3(x, skip3)
        out3 = self.out_dec3(x)  # Deep supervision
        
        x = self.dec2(x, skip2)
        out2 = self.out_dec2(x)  # Deep supervision
        
        x = self.dec1(x, skip1)
        out_final = self.out_final(x)
        
        if return_deep_supervision:
            # Upsample outputs auxiliares al tamaño original
            out2 = F.interpolate(out2, size=(input_size[2], input_size[3]), 
                                mode='bilinear', align_corners=False)
            out3 = F.interpolate(out3, size=(input_size[2], input_size[3]), 
                                mode='bilinear', align_corners=False)
            return out_final, out2, out3
        
        return out_final


class DeepSupervisionLoss(nn.Module):
    """
    Loss con deep supervision para entrenar mejor
    Combina loss de múltiples escalas
    """
    def __init__(self, weights=[1.0, 0.5, 0.25]):
        super().__init__()
        self.weights = weights
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, outputs, target):
        """
        Args:
            outputs: tuple de (out_final, out2, out3)
            target: Ground truth [B, H, W]
        """
        if isinstance(outputs, tuple):
            out_final, out2, out3 = outputs
            
            loss_final = self.criterion(out_final, target)
            loss2 = self.criterion(out2, target)
            loss3 = self.criterion(out3, target)
            
            total_loss = (self.weights[0] * loss_final + 
                         self.weights[1] * loss2 + 
                         self.weights[2] * loss3)
            
            return total_loss
        else:
            return self.criterion(outputs, target)


def count_parameters(model):
    """Cuenta parámetros entrenables del modelo"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Ejemplo de uso
if __name__ == "__main__":
    # Crear modelo
    model = AttentionResUNet(in_channels=3, num_classes=2, base_ch=64)
    
    # Crear tensor de entrada de ejemplo
    x = torch.randn(2, 3, 256, 256)
    
    # Forward pass sin deep supervision
    output = model(x, return_deep_supervision=False)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Forward pass con deep supervision
    out_final, out2, out3 = model(x, return_deep_supervision=True)
    print(f"\nDeep Supervision:")
    print(f"  Final output: {out_final.shape}")
    print(f"  Output 2: {out2.shape}")
    print(f"  Output 3: {out3.shape}")
    
    print(f"\nParámetros totales: {count_parameters(model):,}")
    print(f"Memoria estimada: ~{count_parameters(model) * 4 / (1024**2):.2f} MB")