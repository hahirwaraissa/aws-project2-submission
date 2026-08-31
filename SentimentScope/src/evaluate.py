import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from tqdm import tqdm

def calculate_accuracy(logits, labels):
    """
    Calculates the accuracy of predictions against ground truth labels.
    """
    _, preds = torch.max(logits, dim=1)
    correct = (preds == labels).sum().item()
    return correct / len(labels)

def evaluate_model(model, test_loader, device, model_type="custom", results_dir="results"):
    """
    Evaluates the model on test data, prints/saves classification metrics,
    and plots the confusion matrix.
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    device = torch.device(device)
    model = model.to(device)
    
    print(f"Evaluating {model_type} model on test set...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].cpu().numpy()
            
            logits = model(input_ids, attention_mask=attention_mask)
            _, preds = torch.max(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels)
            
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    
    print("\n" + "="*50)
    print(f"Evaluation Results for {model_type.upper()} Model")
    print("="*50)
    print(f"Accuracy:  {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1-Score:  {f1*100:.2f}%")
    print("="*50)
    
    report = classification_report(all_labels, all_preds, target_names=['Negative', 'Positive'])
    print(report)
    
    # Save metrics text file
    metrics_path = os.path.join(results_dir, f"evaluation_metrics_{model_type}.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"SentimentScope Evaluation Metrics - {model_type.upper()}\n")
        f.write("="*60 + "\n")
        f.write(f"Accuracy:  {accuracy*100:.4f}%\n")
        f.write(f"Precision: {precision*100:.4f}%\n")
        f.write(f"Recall:    {recall*100:.4f}%\n")
        f.write(f"F1-Score:  {f1*100:.4f}%\n")
        f.write("="*60 + "\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        
    print(f"Evaluation report saved to: {metrics_path}")
    
    # Generate and save Confusion Matrix Heatmap
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'],
                annot_kws={"size": 14})
    ax.set_title(f"Confusion Matrix ({model_type.upper()})", color="white", fontsize=14)
    ax.set_xlabel("Predicted Label", color="white")
    ax.set_ylabel("True Label", color="white")
    plt.tight_layout()
    
    cm_path = os.path.join(results_dir, f"confusion_matrix_{model_type}.png")
    fig.savefig(cm_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Confusion matrix plot saved to: {cm_path}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

def plot_training_curves(history, results_dir, model_type="custom"):
    """
    Plots training and validation loss/accuracy curves.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot Loss Curve
    axes[0].plot(epochs, history['train_loss'], label='Train Loss', color='#ff4d4d', marker='o')
    axes[0].plot(epochs, history['val_loss'], label='Val Loss', color='#ffb366', marker='s')
    axes[0].set_title(f"Loss Curves ({model_type.upper()})", color="white")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    
    # Plot Accuracy Curve
    axes[1].plot(epochs, history['train_acc'], label='Train Acc', color='#4dff4d', marker='o')
    axes[1].plot(epochs, history['val_acc'], label='Val Acc', color='#3399ff', marker='s')
    axes[1].set_title(f"Accuracy Curves ({model_type.upper()})", color="white")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    
    plt.tight_layout()
    curve_path = os.path.join(results_dir, f"training_curves_{model_type}.png")
    fig.savefig(curve_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Training history curves saved to: {curve_path}")
