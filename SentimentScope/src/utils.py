import torch
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
from src.preprocessing import clean_text, tokenize

# Set plotting style
sns.set_theme(style="darkgrid")
plt.rcParams.update({'font.size': 12, 'figure.facecolor': '#1e1e1e', 'text.color': '#ffffff', 'axes.labelcolor': '#ffffff', 'xtick.color': '#ffffff', 'ytick.color': '#ffffff'})

def set_seed(seed=42):
    """
    Sets random seeds for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_dirs(base_dir="SentimentScope"):
    """
    Creates standard project directories.
    """
    dirs = [
        os.path.join(base_dir, "data"),
        os.path.join(base_dir, "notebooks"),
        os.path.join(base_dir, "models"),
        os.path.join(base_dir, "results"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return dirs

def load_and_split_data(num_samples=None, seed=42):
    """
    Loads IMDB dataset using Hugging Face datasets library,
    combines default splits, and re-splits into 70% Train, 15% Val, 15% Test.
    Supports loading a smaller subset if num_samples is specified.
    """
    print("Loading IMDB Dataset from Hugging Face...")
    raw_dataset = load_dataset("stanfordnlp/imdb")
    
    # Combine train and test splits to perform standard 70/15/15 split
    df_train = pd.DataFrame(raw_dataset['train'])
    df_test = pd.DataFrame(raw_dataset['test'])
    df = pd.concat([df_train, df_test], ignore_index=True)
    
    # Shuffle the dataset
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    # If subset is requested
    if num_samples is not None and num_samples < len(df):
        print(f"Selecting a subset of {num_samples} samples for quick processing...")
        df = df.head(num_samples)
        
    # Split: 70% train, 30% temp (val + test)
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=seed, stratify=df['label'])
    
    # Split temp: 50% val (15% total), 50% test (15% total)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed, stratify=temp_df['label'])
    
    print(f"Data Splitted - Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df

def run_eda(train_df, results_dir):
    """
    Performs EDA on training dataframe and saves visualization plots.
    """
    print("Running Exploratory Data Analysis (EDA)...")
    
    # 1. Dataset statistics
    total = len(train_df)
    class_counts = train_df['label'].value_counts()
    
    # Compute review token lengths
    print("Computing review lengths...")
    train_df['cleaned_text'] = train_df['text'].apply(clean_text)
    train_df['tokens'] = train_df['cleaned_text'].apply(tokenize)
    train_df['length'] = train_df['tokens'].apply(len)
    
    avg_length = train_df['length'].mean()
    median_length = train_df['length'].median()
    max_length = train_df['length'].max()
    
    print(f"EDA Stats - Total Reviews: {total}")
    print(f"Class Distribution: Negative (0) = {class_counts.get(0, 0)} | Positive (1) = {class_counts.get(1, 0)}")
    print(f"Average Review Length (tokens): {avg_length:.1f} (Median: {median_length:.1f}, Max: {max_length})")
    
    # Save statistics report
    stats_path = os.path.join(results_dir, "eda_statistics.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("SentimentScope - IMDB Training Set Statistics\n")
        f.write("=============================================\n")
        f.write(f"Total reviews in training set: {total}\n")
        f.write(f"Class 0 (Negative) reviews: {class_counts.get(0, 0)} ({class_counts.get(0, 0)/total*100:.1f}%)\n")
        f.write(f"Class 1 (Positive) reviews: {class_counts.get(1, 0)} ({class_counts.get(1, 0)/total*100:.1f}%)\n")
        f.write(f"Average review length: {avg_length:.2f} tokens\n")
        f.write(f"Median review length: {median_length} tokens\n")
        f.write(f"Max review length: {max_length} tokens\n")
    
    # 2. Visualizations
    # Plot A: Class Distribution
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.countplot(data=train_df, x='label', ax=ax, palette=['#ff4d4d', '#4dff4d'])
    ax.set_title("Positive vs Negative Review Counts", color="white")
    ax.set_xticklabels(['Negative (0)', 'Positive (1)'])
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "class_distribution.png"), dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    
    # Plot B: Review Length Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=train_df, x='length', hue='label', kde=True, bins=50, ax=ax, palette=['#ff4d4d', '#4dff4d'], multiple='stack')
    ax.set_title("Review Length Distribution (Word Count)", color="white")
    ax.set_xlabel("Length (Tokens)")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "review_length_distribution.png"), dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    
    # Plot C: Word Clouds and Common Words
    # Basic stopwords list
    stopwords = {"the", "a", "and", "of", "to", "is", "in", "it", "i", "this", "that", "was", "as", "for", "with", "but", "film", "movie", "on", "are", "not", "have", "his", "her", "you", "be", "at", "he", "she", "by", "one", "all", "who", "an", "about", "so", "there", "out", "like", "or", "from", "up", "just", "some", "good", "more", "very", "would", "what", "has", "more"}
    
    pos_words = []
    neg_words = []
    
    for _, row in train_df.iterrows():
        filtered_tokens = [t for t in row['tokens'] if t.isalnum() and t not in stopwords]
        if row['label'] == 1:
            pos_words.extend(filtered_tokens)
        else:
            neg_words.extend(filtered_tokens)
            
    # Common words plots
    pos_counter = Counter(pos_words)
    neg_counter = Counter(neg_words)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Positive common words
    pos_common = pd.DataFrame(pos_counter.most_common(15), columns=['word', 'count'])
    sns.barplot(data=pos_common, x='count', y='word', ax=axes[0], color='#4dff4d')
    axes[0].set_title("Top 15 Positive Movie Words (No Stopwords)", color='white')
    
    # Negative common words
    neg_common = pd.DataFrame(neg_counter.most_common(15), columns=['word', 'count'])
    sns.barplot(data=neg_common, x='count', y='word', ax=axes[1], color='#ff4d4d')
    axes[1].set_title("Top 15 Negative Movie Words (No Stopwords)", color='white')
    
    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "top_common_words.png"), dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    
    # Try importing WordCloud to save Word Clouds
    try:
        from wordcloud import WordCloud
        print("Generating word clouds...")
        
        wordcloud_bg = '#1a1a1a'
        pos_wc = WordCloud(width=800, height=400, background_color=wordcloud_bg, colormap='Greens').generate(" ".join(pos_words))
        neg_wc = WordCloud(width=800, height=400, background_color=wordcloud_bg, colormap='Reds').generate(" ".join(neg_words))
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes[0].imshow(pos_wc, interpolation='bilinear')
        axes[0].axis('off')
        axes[0].set_title("Positive Reviews Word Cloud", color='white', fontsize=18)
        
        axes[1].imshow(neg_wc, interpolation='bilinear')
        axes[1].axis('off')
        axes[1].set_title("Negative Reviews Word Cloud", color='white', fontsize=18)
        
        plt.tight_layout()
        fig.savefig(os.path.join(results_dir, "wordclouds.png"), dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)
        print("Word clouds saved successfully.")
    except ImportError:
        print("WordCloud package not available. Skipping word cloud generation.")
        
    print(f"EDA visualizations saved in: {results_dir}")
