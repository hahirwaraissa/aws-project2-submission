import argparse
import os
import torch
import numpy as np
import pandas as pd
import json

from src.utils import set_seed, create_dirs, load_and_split_data, run_eda
from src.preprocessing import clean_text, get_bert_tokenizer
from src.dataset import IMDBDataset, get_dataloaders
from src.model import CustomTransformerClassifier, DistilBertClassifier, DemoGPT
from src.train import train_model, get_device
from src.evaluate import evaluate_model, plot_training_curves

def predict_sentiment(text, model, tokenizer, max_len, device):
    """
    Predicts the sentiment of a single review text using the subword tokenizer.
    Returns the predicted class (Positive/Negative) and the confidence percentage.
    """
    model.eval()
    
    # 1. Clean
    cleaned = clean_text(text)
    
    # 2. Tokenize and pad using BERT subword tokenizer
    encoding = tokenizer(
        cleaned,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
        
    # Forward pass
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
    pred_class = np.argmax(probs)
    sentiment = "Positive" if pred_class == 1 else "Negative"
    confidence = probs[pred_class]
    
    return sentiment, confidence

def main():
    parser = argparse.ArgumentParser(description="SentimentScope - Transformer Sentiment Classifier")
    parser.add_argument('--mode', type=str, required=True, choices=['eda', 'train', 'evaluate', 'predict', 'onnx'],
                        help="Execution mode")
    parser.add_argument('--model_type', type=str, default='custom', choices=['custom', 'distilbert'],
                        help="Model architecture: scratch Custom Transformer vs pre-trained DistilBERT")
    
    # Training args
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Mini-batch size")
    parser.add_argument('--lr', type=float, default=2e-4, help="Learning rate")
    parser.add_argument('--patience', type=int, default=3, help="Early stopping patience")
    parser.add_argument('--num_samples', type=int, default=None, 
                        help="Number of reviews to load (None=all 50k, 1000=quick subset for CPU)")
    
    # Model configuration args
    parser.add_argument('--max_len', type=int, default=256, help="Maximum sequence length")
    parser.add_argument('--embed_dim', type=int, default=128, help="Embedding dimension (custom model)")
    parser.add_argument('--num_heads', type=int, default=4, help="Attention heads (custom model)")
    parser.add_argument('--num_layers', type=int, default=2, help="Encoder layers (custom model)")
    parser.add_argument('--feedforward_dim', type=int, default=256, help="FFN dimension (custom model)")
    parser.add_argument('--dropout', type=float, default=0.1, help="Dropout rate")
    
    # Prediction arg
    parser.add_argument('--text', type=str, default=None, help="Text to predict sentiment for")
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Define directories
    base_dir = "SentimentScope"
    data_dir, notebooks_dir, models_dir, results_dir = create_dirs(base_dir)
    
    device = get_device()
    
    # Instantiate the BERT subword tokenizer (used for both models)
    tokenizer = get_bert_tokenizer('bert-base-uncased')
    
    if args.mode == 'eda':
        # Load data and run EDA
        train_df, _, _ = load_and_split_data(num_samples=args.num_samples)
        run_eda(train_df, results_dir)
        
    elif args.mode == 'train':
        # Load and split
        train_df, val_df, test_df = load_and_split_data(num_samples=args.num_samples)
        
        # Model configuration dictionary
        model_config = {
            'vocab_size': tokenizer.vocab_size,
            'embed_dim': args.embed_dim,
            'num_heads': args.num_heads,
            'num_layers': args.num_layers,
            'feedforward_dim': args.feedforward_dim,
            'dropout': args.dropout,
            'max_seq_len': args.max_len,
            'model_type': args.model_type
        }
        
        # Save custom model configuration
        if args.model_type == 'custom':
            config_path = os.path.join(models_dir, "model_config_custom.json")
            with open(config_path, 'w') as f:
                json.dump(model_config, f, indent=4)
            
            # Setup Model from scratch (DemoGPT class)
            model = DemoGPT(
                vocab_size=tokenizer.vocab_size,
                embed_dim=args.embed_dim,
                num_heads=args.num_heads,
                num_layers=args.num_layers,
                feedforward_dim=args.feedforward_dim,
                dropout=args.dropout,
                max_seq_len=args.max_len
            )
        else: # distilbert
            # Setup Model utilizing pre-trained DistilBERT weights
            model = DistilBertClassifier(num_classes=2, dropout=args.dropout)
            
        # Setup Datasets using subword tokenizer
        train_dataset = IMDBDataset(train_df['text'].tolist(), train_df['label'].tolist(), tokenizer, max_len=args.max_len)
        val_dataset = IMDBDataset(val_df['text'].tolist(), val_df['label'].tolist(), tokenizer, max_len=args.max_len)
        test_dataset = IMDBDataset(test_df['text'].tolist(), test_df['label'].tolist(), tokenizer, max_len=args.max_len)
        
        # Get PyTorch loaders
        train_loader, val_loader, test_loader = get_dataloaders(
            train_dataset, val_dataset, test_dataset, batch_size=args.batch_size
        )
        
        # Train model
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            vocab=None,  # No custom vocab object needed anymore since we use HF tokenizer
            model_config=model_config,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            checkpoint_dir=models_dir,
            model_type=args.model_type
        )
        
        # Plot curves
        plot_training_curves(history, results_dir, model_type=args.model_type)
        
        # Load best model for evaluation on test set
        checkpoint_path = os.path.join(models_dir, f"best_sentimentscope_{args.model_type}.pt")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate model on test set
        evaluate_model(model, test_loader, device=device, model_type=args.model_type, results_dir=results_dir)
        
    elif args.mode == 'evaluate':
        # Setup evaluation
        checkpoint_path = os.path.join(models_dir, f"best_sentimentscope_{args.model_type}.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}. Please train the model first.")
            
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load splits
        _, _, test_df = load_and_split_data(num_samples=args.num_samples)
        
        if args.model_type == 'custom':
            model = DemoGPT(
                vocab_size=tokenizer.vocab_size,
                embed_dim=checkpoint['model_config']['embed_dim'],
                num_heads=checkpoint['model_config']['num_heads'],
                num_layers=checkpoint['model_config']['num_layers'],
                feedforward_dim=checkpoint['model_config']['feedforward_dim'],
                dropout=checkpoint['model_config']['dropout'],
                max_seq_len=checkpoint['model_config']['max_seq_len']
            )
        else: # distilbert
            model = DistilBertClassifier(num_classes=2)
            
        model.load_state_dict(checkpoint['model_state_dict'])
        
        test_dataset = IMDBDataset(test_df['text'].tolist(), test_df['label'].tolist(), tokenizer, max_len=args.max_len)
            
        # Get loader
        _, _, test_loader = get_dataloaders(test_dataset, test_dataset, test_dataset, batch_size=args.batch_size)
        
        # Evaluate
        evaluate_model(model, test_loader, device=device, model_type=args.model_type, results_dir=results_dir)
        
    elif args.mode == 'predict':
        checkpoint_path = os.path.join(models_dir, f"best_sentimentscope_{args.model_type}.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}. Please train the model first.")
            
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model architecture
        if args.model_type == 'custom':
            model = DemoGPT(
                vocab_size=tokenizer.vocab_size,
                embed_dim=checkpoint['model_config']['embed_dim'],
                num_heads=checkpoint['model_config']['num_heads'],
                num_layers=checkpoint['model_config']['num_layers'],
                feedforward_dim=checkpoint['model_config']['feedforward_dim'],
                dropout=checkpoint['model_config']['dropout'],
                max_seq_len=checkpoint['model_config']['max_seq_len']
            )
        else: # distilbert
            model = DistilBertClassifier(num_classes=2)
            
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        if args.text:
            sentiment, conf = predict_sentiment(args.text, model, tokenizer, args.max_len, device)
            print("\n" + "-"*40)
            print(f"Review: \"{args.text}\"")
            print(f"Prediction: {sentiment}")
            print(f"Confidence: {conf * 100:.1f}%")
            print("-"*40 + "\n")
        else:
            # Run multiple pre-defined test examples
            test_examples = [
                "This movie was absolutely amazing! The story was gripping and the acting was top notch.",
                "Worst film I have seen in years. Terribly written, boring, and a complete waste of time.",
                "I had high expectations, but it turned out to be just average. Nothing spectacular.",
                "An absolute masterpiece. Beautiful cinematography and a deep, emotional musical score.",
                "Very disappointing. The plot was full of holes and the characters were unlikable."
            ]
            print("\n" + "="*70)
            print("SentimentScope Inference - Running Test Examples")
            print("="*70)
            for idx, text in enumerate(test_examples, 1):
                sentiment, conf = predict_sentiment(text, model, tokenizer, args.max_len, device)
                print(f"[{idx}] Text: \"{text}\"")
                print(f"    Prediction: {sentiment} (Confidence: {conf*100:.1f}%)")
                print("-" * 70)
                
    elif args.mode == 'onnx':
        if args.model_type != 'custom':
            raise NotImplementedError("ONNX export is currently supported only for the custom model from scratch.")
            
        checkpoint_path = os.path.join(models_dir, "best_sentimentscope_custom.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Custom model checkpoint not found at: {checkpoint_path}. Please train the model first.")
            
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model architecture
        model = DemoGPT(
            vocab_size=tokenizer.vocab_size,
            embed_dim=checkpoint['model_config']['embed_dim'],
            num_heads=checkpoint['model_config']['num_heads'],
            num_layers=checkpoint['model_config']['num_layers'],
            feedforward_dim=checkpoint['model_config']['feedforward_dim'],
            dropout=checkpoint['model_config']['dropout'],
            max_seq_len=checkpoint['model_config']['max_seq_len']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.to(device)
        
        # Export settings
        onnx_path = os.path.join(models_dir, "sentimentscope.onnx")
        
        # Create dummy inputs: (batch_size=1, max_len)
        dummy_input_ids = torch.zeros((1, args.max_len), dtype=torch.long, device=device)
        dummy_attention_mask = torch.ones((1, args.max_len), dtype=torch.long, device=device)
        
        print(f"Tracing model and exporting to ONNX format...")
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_attention_mask),
            onnx_path,
            input_names=['input_ids', 'attention_mask'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size'},
                'attention_mask': {0: 'batch_size'},
                'logits': {0: 'batch_size'}
            },
            opset_version=14,
            do_constant_folding=True
        )
        
        # Verify ONNX model exists
        if os.path.exists(onnx_path):
            print(f"✓ ONNX model successfully saved to: {onnx_path}")
            
            # Test inference using onnxruntime
            import onnxruntime as ort
            print("Verifying ONNX model using onnxruntime...")
            ort_session = ort.InferenceSession(onnx_path)
            
            # Prepare input
            np_ids = dummy_input_ids.cpu().numpy()
            np_mask = dummy_attention_mask.cpu().numpy()
            
            # Feed inputs
            ort_inputs = {
                'input_ids': np_ids,
                'attention_mask': np_mask
            }
            ort_outs = ort_session.run(None, ort_inputs)
            print(f"✓ ONNX verification output shape: {ort_outs[0].shape}")
            print(f"✓ ONNX Export complete.")
        else:
            print("✗ Error exporting to ONNX.")

if __name__ == '__main__':
    main()
