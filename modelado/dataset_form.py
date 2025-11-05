import torch
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Transformers
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Formación de Data
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, tile_ids, images_dir, annotations_dict, transform=None):
        self.tile_ids = tile_ids
        self.images_dir = Path(images_dir)
        self.annotations_dict = annotations_dict
        self.transform = transform
    
    def __len__(self):
        return len(self.tile_ids)
    
    def __getitem__(self, idx):
        tile_id = self.tile_ids[idx]
        
        # Cargar imagen
        img = cv2.imread(str(self.images_dir / f"{tile_id}.tif"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Crear máscara
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        if tile_id in self.annotations_dict:
            for ann in self.annotations_dict[tile_id]['annotations']:
                if ann['type'] == 'blood_vessel':
                    coords = np.array(ann['coordinates'], dtype=np.int32)
                    cv2.fillPoly(mask, [coords], 1)
                elif ann['type'] == 'glomerulus':
                    coords = np.array(ann['coordinates'], dtype=np.int32)
                    cv2.fillPoly(mask, [coords], 255)  # Ignorar
        
        # Augmentation
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        return img, mask.long(), self.tile_ids
