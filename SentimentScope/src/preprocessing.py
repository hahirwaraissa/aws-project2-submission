import re
from transformers import AutoTokenizer

def clean_text(text):
    """
    Cleans movie review text by:
    1. Converting to lowercase.
    2. Removing HTML tags like <br />.
    3. Standardizing whitespace.
    """
    if not isinstance(text, str):
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]*>', ' ', text)
    
    # Lowercase
    text = text.lower()
    
    # Standardize whitespace (replace multiple spaces/newlines with single space)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def tokenize(text):
    """
    Splits text into tokens using a simple regex that captures words and punctuation separately.
    Used mainly for EDA word count statistics.
    """
    return re.findall(r'\w+|[^\w\s]', text)

def get_bert_tokenizer(model_name="bert-base-uncased"):
    """
    Loads and returns the pre-trained BERT subword tokenizer.
    """
    print(f"Loading pre-trained subword tokenizer: {model_name}...")
    return AutoTokenizer.from_pretrained(model_name)
