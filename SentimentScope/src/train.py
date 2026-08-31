import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import numpy as np

def get_device():
    """
    Returns the best available hardware accelerator:
    1. CUDA GPU (Nvidia)
    2. MPS (Apple Silicon GPU)
    3. CPU (fallback)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Saves the best model checkpoint automatically.
    """
    def __init__(self, patience=3, verbose=True, delta=0, path='best_model.pt'):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model, optimizer, epoch, vocab=None, model_config=None):
        score = -val_loss

        if self.best_loss is None:
            self.best_loss = score
            self.save_checkpoint(val_loss, model, optimizer, epoch, vocab, model_config)
        elif score < self.best_loss + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = score
            self.save_checkpoint(val_loss, model, optimizer, epoch, vocab, model_config)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer, epoch, vocab, model_config):
        """
        Saves model state dictionary when validation loss decreases.
        """
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model checkpoint...")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'model_config': model_config
        }
        
        if vocab is not None:
            # We can serialise the vocabulary metadata here for custom model
            checkpoint['vocab_stoi'] = vocab.stoi
            checkpoint['vocab_itos'] = vocab.itos
            
        torch.save(checkpoint, self.path)
        self.val_loss_min = val_loss


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    # Progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    for batch in pbar:
        # Move inputs to target device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(input_ids, attention_mask=attention_mask)
        
        # Calculate loss
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients in Transformers
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        # Compute accuracy and aggregate loss
        total_loss += loss.item() * len(labels)
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels).item()
        total_samples += len(labels)
        
        # Update progress bar description
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{(correct_predictions / total_samples) * 100:.2f}%"
        })
        
    epoch_loss = total_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    
    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device, epoch):
    """
    Validates the model for one epoch.
    """
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
    
    with torch.no_grad():
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            logits = model(input_ids, attention_mask=attention_mask)
            
            # Loss
            loss = criterion(logits, labels)
            
            # Compute accuracy and loss
            total_loss += loss.item() * len(labels)
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels).item()
            total_samples += len(labels)
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{(correct_predictions / total_samples) * 100:.2f}%"
            })
            
    epoch_loss = total_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    
    return epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader, vocab, model_config,
                epochs=10, lr=1e-4, patience=3, checkpoint_dir="models",
                tensorboard_dir="results/tensorboard", model_type="custom"):
    """
    Main training orchestrator for custom and Hugging Face models.
    """
    device = get_device()
    model = model.to(device)
    
    # 1. Setup loss, optimizer, scheduler, early stopping
    criterion = nn.CrossEntropyLoss()
    
    # Use different learning rates for custom vs pretrained transformer
    # Pretrained models require a lower learning rate (e.g. 2e-5) to avoid catastrophic forgetting
    if model_type == "distilbert":
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        
    # Learning rate scheduler (decays LR if validation loss plateaus)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1
    )
    
    # Create save directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"best_sentimentscope_{model_type}.pt")
    
    early_stopping = EarlyStopping(
        patience=patience, 
        verbose=True, 
        path=checkpoint_path
    )
    
    # Tensorboard logger
    writer = SummaryWriter(log_dir=os.path.join(tensorboard_dir, model_type))
    
    # History logs for return
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    print(f"Starting training for {model_type} model...")
    for epoch in range(1, epochs + 1):
        # Train one epoch
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate one epoch
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device, epoch)
        
        # Log to TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Val', val_acc, epoch)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
        
        # Save to history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch} Summary - Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
        
        # Scheduler step based on validation loss
        scheduler.step(val_loss)
        
        # Early Stopping check
        early_stopping(val_loss, model, optimizer, epoch, vocab, model_config)
        if early_stopping.early_stop:
            print("Early stopping triggered. Training stopped.")
            break
            
    writer.close()
    print(f"Training completed. Best model checkpoint saved to: {checkpoint_path}")
    return history
