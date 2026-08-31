import torch
from torch.utils.data import Dataset, DataLoader
from src.preprocessing import clean_text

class IMDBDataset(Dataset):
    """
    Custom PyTorch Dataset for Transformer sentiment classifier.
    Preprocesses raw text using clean_text and encodes it using the pre-trained subword tokenizer.
    """
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Clean text
        cleaned = clean_text(text)

        # Encode using the BERT-based subword tokenizer
        encoding = self.tokenizer(
            cleaned,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=32, num_workers=0):
    """
    Creates and returns Train, Validation, and Test DataLoaders.
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader
