import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import os
from pathlib import Path
from tqdm import tqdm

from models import DynamicCompactDetect
from utils.datasets import COCODataset
from utils.detection_loss import YOLOLoss


def parse_args():
    parser = argparse.ArgumentParser(description='Train DynamicCompact-Detect model')
    parser.add_argument('--cfg', type=str, default='configs/dynamiccompact_minimal.yaml', help='model configuration file')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='data configuration file')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='batch size')
    parser.add_argument('--img-size', type=int, default=640, help='input image size')
    parser.add_argument('--output', type=str, default='runs/train_minimal', help='output folder')
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(output_dir / 'args.yaml', 'w') as f:
        yaml.dump(vars(args), f)
    
    # Load data configuration
    with open(args.data) as f:
        data_dict = yaml.safe_load(f)
    
    # Load model configuration
    with open(args.cfg) as f:
        model_dict = yaml.safe_load(f)
    
    # Create model
    model = DynamicCompactDetect(args.cfg).to(device)
    print(f'Model created with {args.cfg}')
    
    # Save a copy of the model configuration
    with open(output_dir / 'model_config.yaml', 'w') as f:
        yaml.dump(model_dict, f)
    
    # Create datasets
    train_dataset = COCODataset(
        data_dict['train'], 
        img_size=args.img_size,
        augment=True,
        rect=False,
        batch_size=args.batch_size
    )
    
    val_dataset = COCODataset(
        data_dict['val'],
        img_size=args.img_size,
        augment=False,
        rect=True,
        batch_size=args.batch_size
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=val_dataset.collate_fn
    )
    
    # Create loss function
    anchors = model_dict['anchors']
    loss_fn = YOLOLoss(anchors=anchors, num_classes=model_dict['nc'], img_size=args.img_size)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        
        for i, (imgs, targets, _) in enumerate(pbar):
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            preds = model(imgs)
            
            # Calculate loss
            loss_dict = loss_fn(preds, targets)
            loss = loss_dict['loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': train_loss / (i + 1)})
            
            # Only process a small subset for quick demonstration
            if i >= 50:
                break
        
        # Calculate average loss
        avg_train_loss = train_loss / min(50, len(train_loader))
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(val_loader, desc='Validating')
        
        with torch.no_grad():
            for i, (imgs, targets, _) in enumerate(val_pbar):
                imgs = imgs.to(device)
                targets = targets.to(device)
                
                # Forward pass
                preds = model(imgs)
                
                # Calculate loss
                loss_dict = loss_fn(preds, targets)
                loss = loss_dict['loss']
                
                # Update metrics
                val_loss += loss.item()
                
                # Only process a small subset for quick demonstration
                if i >= 20:
                    break
        
        # Calculate average validation loss
        avg_val_loss = val_loss / min(20, len(val_loader))
        
        # Print metrics
        print(f'Epoch {epoch+1}/{args.epochs}, '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}')
        
        # Save checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            # Save the model configuration
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
                'config_path': args.cfg
            }
            
            torch.save(checkpoint, output_dir / 'best_model.pt')
            print(f'Model saved to {output_dir}/best_model.pt')
    
    # Save final model
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_val_loss,
        'config_path': args.cfg
    }, output_dir / 'final_model.pt')
    
    print(f'Training completed and model saved to {output_dir}/final_model.pt')


if __name__ == '__main__':
    main() 