import torch
import numpy as np
import os
from transformers import AutoTokenizer
from src.model import DemoGPT
from src.preprocessing import clean_text

class SentimentClassifierInterface:
    """
    Inference interface class for SentimentScope (DemoGPT).
    Loads a saved model checkpoint and performs batch inference on a list of texts.
    """
    def __init__(self, checkpoint_path, tokenizer_name="bert-base-uncased", device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"Initializing tokenizer: {tokenizer_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
            
        print(f"Loading weights from checkpoint: {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint['model_config']
        
        self.max_len = config.get('max_seq_len', 256)
        
        # Instantiate DemoGPT model using the config saved at training time
        self.model = DemoGPT(
            vocab_size=config['vocab_size'],
            embed_dim=config['embed_dim'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            feedforward_dim=config['feedforward_dim'],
            dropout=config['dropout'],
            max_seq_len=self.max_len
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded and set to evaluation mode successfully.")

    def predict(self, texts):
        """
        Performs sentiment prediction on a batch (list) of string reviews.
        Returns a list of dictionaries with predictions and probability confidence.
        """
        if isinstance(texts, str):
            texts = [texts]
            
        results = []
        for text in texts:
            # Clean and tokenize
            cleaned = clean_text(text)
            encoding = self.tokenizer(
                cleaned,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Predict
            with torch.no_grad():
                logits = self.model(input_ids, attention_mask=attention_mask)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                
            pred_class = np.argmax(probs)
            label = "Positive" if pred_class == 1 else "Negative"
            confidence = probs[pred_class]
            
            results.append({
                "text": text,
                "label": label,
                "confidence": float(confidence)
            })
            
        return results
