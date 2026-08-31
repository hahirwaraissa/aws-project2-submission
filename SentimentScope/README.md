# SentimentScope: Transformer-Based Sentiment Analysis System

SentimentScope is a production-quality, end-to-end sentiment classification system developed for **CineScope** to analyze IMDB movie reviews and classify them into **Positive (1)** or **Negative (0)** sentiments. 

This repository contains two main model architectures:
1. **Custom Transformer Classifier (Built from Scratch)**: A pure PyTorch implementation of the Transformer encoder architecture, featuring custom multi-head self-attention, positional encoding, and masked global average pooling.
2. **Hugging Face DistilBERT Baseline**: A pre-trained transformer wrapper utilized for benchmarking and performance comparison.

---

## 📂 Project Structure

```text
SentimentScope/
├── data/                       # Cached datasets
├── notebooks/                  # Interactive experimentation sandboxes
├── models/                     # Saved checkpoints, vocabularies, and ONNX files
│   ├── best_sentimentscope_custom.pt   # Scratch PyTorch model weights
│   ├── vocab.json                      # Custom word-to-index vocabulary
│   ├── model_config_custom.json        # Custom model architecture settings
│   ├── sentimentscope.onnx             # Exported ONNX format graph
│   └── sentimentscope.onnx.data        # ONNX tensor weights data
├── results/                    # Saved plots and performance reports
│   ├── class_distribution.png          # Train set class balance plot
│   ├── review_length_distribution.png  # Review sequence length histogram
│   ├── top_common_words.png            # Common word frequencies (no stopwords)
│   ├── wordclouds.png                  # Positive & negative word clouds
│   ├── training_curves_custom.png      # Loss/Accuracy progression
│   ├── confusion_matrix_custom.png     # Test set confusion matrix heatmap
│   ├── eda_statistics.txt              # Text and length statistics
│   └── evaluation_metrics_custom.txt   # Classification report
├── src/                        # Modular source package
│   ├── dataset.py              # PyTorch Dataset and DataLoader generators
│   ├── preprocessing.py        # Cleaning, Tokenization, Vocab building, and Padding
│   ├── model.py                # Scratch Transformer and HF DistilBERT models
│   ├── train.py                # Training loop, early stopping, and schedulers
│   ├── evaluate.py             # Validation loops and metric visualizations
│   └── utils.py                # Seeds, directories, and EDA utilities
├── requirements.txt            # Dependency configuration
├── README.md                   # System documentation
└── main.py                     # Execution orchestrator CLI
```

---

## 🧠 Model Architecture (Custom Transformer)

The custom model is implemented from first principles in `src/model.py` and consists of:

### 1. Embeddings & Positional Encoding
- **Token Embeddings**: Maps discrete input token indices to continuous vectors using `nn.Embedding`.
- **Positional Encoding**: Injects structural sequence order information using sinusoidal waves:
  $$\text{PE}_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
  $$\text{PE}_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

### 2. Multi-Head Self-Attention (MHA)
Our custom MHA projects input representations into Queries ($Q$), Keys ($K$), and Values ($V$) across multiple independent attention heads. It computes scaled dot-product attention while supporting attention masks to prevent processing `<pad>` tokens:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$
Where $M$ is the attention mask ($0$ for real tokens, $-1\times 10^9$ for padded index positions).

### 3. Pre-LN Encoder Layers
To improve gradient stability and training convergence, we implement the **Pre-LayerNorm (Pre-LN)** Transformer layer structure:
$$\mathbf{h}^{(1)} = \mathbf{x} + \text{Dropout}(\text{MHA}(\text{LN}(\mathbf{x})))$$
$$\mathbf{x}_{\text{out}} = \mathbf{h}^{(1)} + \text{Dropout}(\text{FFN}(\text{LN}(\mathbf{h}^{(1)})))$$
Where $\text{FFN}$ is a two-layer feedforward network with GELU non-linear activations.

### 4. Masked Global Average Pooling
Instead of standard pooling which introduces bias from padding tokens, SentimentScope performs **Masked Global Average Pooling**, dividing the sum of non-padded token vectors by the active sequence length:
$$\mathbf{p} = \frac{\sum_{t=1}^{T} \mathbf{x}_t \cdot \mathbb{1}(token_t \neq \langle pad \rangle)}{\sum_{t=1}^{T} \mathbb{1}(token_t \neq \langle pad \rangle)}$$

### 5. Classification Head
Outputs probability logits over 2 target classes (Negative, Positive) through a dense linear layer configuration.

---

## ⚡ Installation

Install dependencies from the root directory:
```bash
pip install -r SentimentScope/requirements.txt
```

---

## 🚀 Execution & Usage

The system is fully controlled via `main.py` command line arguments.

### 1. Exploratory Data Analysis (EDA)
Downloads the dataset and saves plots (word clouds, distribution checks) to `results/`:
```bash
python SentimentScope/main.py --mode eda --num_samples 1000
```
*(Omit `--num_samples` to run on the entire 50,000 review IMDB dataset).*

### 2. Model Training & Evaluation
Trains the custom transformer, saves checkpoints, outputs training loss/accuracy curves, and evaluates on the test split:
```bash
python SentimentScope/main.py --mode train --num_samples 1000 --epochs 5 --batch_size 16 --lr 2e-4
```
For pre-trained DistilBERT:
```bash
python SentimentScope/main.py --mode train --model_type distilbert --num_samples 1000 --epochs 3 --batch_size 8
```

### 3. Evaluate an Existing Checkpoint
Runs model evaluation on the test split using a pre-saved checkpoint:
```bash
python SentimentScope/main.py --mode evaluate --model_type custom
```

### 4. Single Sentence and Batch Inference
Classifies sentiment on a custom review text:
```bash
python SentimentScope/main.py --mode predict --model_type custom --text "This movie was absolutely amazing! The story was gripping."
```
To run prediction on standard validation test cases, omit the `--text` argument:
```bash
python SentimentScope/main.py --mode predict --model_type custom
```

### 5. Model Serialization (ONNX Export)
Traces and serializes the Custom Transformer to ONNX format for high-performance deployments:
```bash
python SentimentScope/main.py --mode onnx
```

---

## 📊 Visual Performance Report & Metrics

After training (e.g., on a 1,000-sample test run), the outputs are saved under `results/`:
* **`eda_statistics.txt`**: Logs review token statistics (Mean, Median, Max length) and class counts.
* **`class_distribution.png`**: Visualizes balance between labels.
* **`review_length_distribution.png`**: Shows word lengths per sentiment.
* **`top_common_words.png`**: Displays key non-stop words.
* **`wordclouds.png`**: Highlights positive vs negative semantic vocab.
* **`training_curves_custom.png`**: Plots loss/accuracy convergence over training epochs.
* **`confusion_matrix_custom.png`**: Shows true vs predicted sentiment mapping.

---

## 🔮 Future Improvements

1. **Subword Tokenization**: Replace the word-level tokenizer with a Byte-Pair Encoding (BPE) or WordPiece tokenizer.
2. **Hyperparameter Tuning**: Integrate Ray Tune or Optuna to optimize learning rates, dropout rates, and layer depths.
3. **Quantization**: Quantize model weights from float32 to int8 to reduce ONNX latency.
4. **API Integration**: Wrap the ONNX model in a FastAPI container for real-time recommendation system requests.
