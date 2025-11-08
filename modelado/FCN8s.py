import torch.nn as nn
from torchvision import models

class FCN8s(nn.Module):
    """
    FCN-8s para segmentación semántica
    Usa VGG16 como backbone + decoders progresivos
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(FCN8s, self).__init__()
        
        # Cargar VGG16 preentrenado
        vgg = models.vgg16(pretrained=pretrained)
        features = list(vgg.features.children())
        
        # Dividir VGG en bloques
        self.features_block1 = nn.Sequential(*features[:5])   # Pool 1
        self.features_block2 = nn.Sequential(*features[5:10])  # Pool 2
        self.features_block3 = nn.Sequential(*features[10:17]) # Pool 3
        self.features_block4 = nn.Sequential(*features[17:24]) # Pool 4
        self.features_block5 = nn.Sequential(*features[24:])   # Pool 5
        
        # Reemplazar FC por convoluciones (el "fully convolutional")
        self.fc6 = nn.Conv2d(512, 4096, kernel_size=7, padding=3)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()
        
        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()
        
        # Score layers (clasificación)
        self.score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
        
        # Upsampling (deconvolución)
        self.upscore2 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=4, stride=2, bias=False
        )
        self.upscore8 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=16, stride=8, bias=False
        )
        self.upscore_pool4 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=4, stride=2, bias=False
        )
        
        # Inicializar pesos de upsampling
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicialización bilinear para upsampling"""
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        input_size = x.size()
        
        # Encoder (VGG16)
        pool3 = self.features_block3(self.features_block2(self.features_block1(x)))
        pool4 = self.features_block4(pool3)
        pool5 = self.features_block5(pool4)
        
        # Fully convolutional layers
        h = self.relu6(self.fc6(pool5))
        h = self.drop6(h)
        h = self.relu7(self.fc7(h))
        h = self.drop7(h)
        
        # Score (clasificación)
        h = self.score_fr(h)
        h = self.upscore2(h)
        
        # Fusión con pool4 (ajustar tamaños dinámicamente)
        score_pool4 = self.score_pool4(pool4)
        
        # Crop h para que coincida con score_pool4
        diff_h = h.size(2) - score_pool4.size(2)
        diff_w = h.size(3) - score_pool4.size(3)
        
        h = h[:, :, 
              diff_h//2:diff_h//2 + score_pool4.size(2),
              diff_w//2:diff_w//2 + score_pool4.size(3)]
        
        h = h + score_pool4
        h = self.upscore_pool4(h)
        
        # Fusión con pool3 (ajustar tamaños dinámicamente)
        score_pool3 = self.score_pool3(pool3)
        
        # Crop h para que coincida con score_pool3
        diff_h = h.size(2) - score_pool3.size(2)
        diff_w = h.size(3) - score_pool3.size(3)
        
        h = h[:, :,
              diff_h//2:diff_h//2 + score_pool3.size(2),
              diff_w//2:diff_w//2 + score_pool3.size(3)]
        
        h = h + score_pool3
        
        # Upsample final a tamaño original
        h = self.upscore8(h)
        
        # Crop al tamaño de entrada original
        diff_h = h.size(2) - input_size[2]
        diff_w = h.size(3) - input_size[3]
        
        h = h[:, :,
              diff_h//2:diff_h//2 + input_size[2],
              diff_w//2:diff_w//2 + input_size[3]]
        
        return h