import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
import time
from tqdm import tqdm
from modeloC_tiny import AttentionUNetTiny
from dataset_form import SimpleDataset, train_transform, val_transform

def dice_score(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred) if pred.shape[1] == 1 else torch.softmax(pred, dim=1)[:, 1:2]
    pred = (pred > 0.5).float()
    target = (target == 1).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def iou_score(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred) if pred.shape[1] == 1 else torch.softmax(pred, dim=1)[:, 1:2]
    pred = (pred > 0.5).float()
    target = (target == 1).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def precision_recall_f1(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred) if pred.shape[1] == 1 else torch.softmax(pred, dim=1)[:, 1:2]
    pred = (pred > 0.5).float()
    target = (target == 1).float()
    
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    f1 = 2 * precision * recall / (precision + recall + smooth)
    
    return precision.item(), recall.item(), f1.item()

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for images, masks, _ in tqdm(loader, desc="Training"):
        images, masks = images.to(device), masks.to(device).long()
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    dice_scores, iou_scores = [], []
    precisions, recalls, f1_scores = [], [], []
    
    with torch.no_grad():
        for images, masks, _ in tqdm(loader, desc="Validating"):
            images, masks = images.to(device), masks.to(device).long()
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            
            dice = dice_score(outputs, masks)
            iou = iou_score(outputs, masks)
            prec, rec, f1 = precision_recall_f1(outputs, masks)
            
            dice_scores.append(dice.item())
            iou_scores.append(iou.item())
            precisions.append(prec)
            recalls.append(rec)
            f1_scores.append(f1)
    
    return {
        'loss': total_loss / len(loader),
        'dice': sum(dice_scores) / len(dice_scores),
        'iou': sum(iou_scores) / len(iou_scores),
        'precision': sum(precisions) / len(precisions),
        'recall': sum(recalls) / len(recalls),
        'f1': sum(f1_scores) / len(f1_scores)
    }

def main():
    # Config
    device = torch.device('cpu')
    num_epochs = 5
    batch_size = 8
    learning_rate = 1e-4
    checkpoint_path = 'models/best_attention_unet_tiny.pth'
    
    print(f"Using device: {device}")
    
    # Load data
    with open('modelado/data_minimal_filtrado.pkl', 'rb') as f:
        data = pickle.load(f)
    
    train_dataset = SimpleDataset(data['train'], 'train/', data['annotations'], train_transform)
    val_dataset = SimpleDataset(data['val'], 'train/', data['annotations'], val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Model
    model = AttentionUNetTiny(in_channels=3, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_dice': [],
        'val_iou': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'epoch_times': []
    }
    
    best_dice = 0
    start_epoch = 0
    
    # ===== CARGAR CHECKPOINT SI EXISTE =====
    import os
    if os.path.exists(checkpoint_path):
        print(f"\n🔄 Cargando checkpoint desde {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        best_dice = checkpoint['dice_score']
        history = checkpoint['history']
        print(f"✓ Continuando desde época {start_epoch} (Best Dice: {best_dice:.4f})")
    else:
        print("\n🆕 Entrenamiento nuevo desde cero")
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{start_epoch + num_epochs}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - start_time
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_dice'].append(val_metrics['dice'])
        history['val_iou'].append(val_metrics['iou'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])
        history['epoch_times'].append(epoch_time)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val Dice: {val_metrics['dice']:.4f}")
        print(f"Val IoU: {val_metrics['iou']:.4f}")
        print(f"Val Precision: {val_metrics['precision']:.4f}")
        print(f"Val Recall: {val_metrics['recall']:.4f}")
        print(f"Val F1: {val_metrics['f1']:.4f}")
        print(f"Epoch Time: {epoch_time:.2f}s")
        
        # Save best model
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'dice_score': best_dice,
                'history': history
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"✓ Best model saved! Dice: {best_dice:.4f}")
    
    print(f"\n{'='*60}")
    print(f"Training Complete! Best Dice: {best_dice:.4f}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()