# SentimentScope: End-to-End Transformer Sentiment Classifier

SentimentScope is a production-ready, Transformer-based binary classification system designed to categorize IMDB movie reviews into **Positive (1)** or **Negative (0)** sentiments. 

This repository contains:
1. **DemoGPT**: A custom Transformer encoder architecture built from scratch in PyTorch, featuring Sinusoidal Positional Encoding, Multi-Head Self-Attention, GELU FeedForward networks, and Masked Global Average Pooling.
2. **Hugging Face DistilBERT Baseline**: A pre-trained transformer wrapper utilized for benchmarking and performance comparison.
3. **Batch Inference Interface**: A user-friendly API class to load model weights and perform bulk predictions on batches of reviews.
4. **ONNX Compiler Graph**: A compiled version of the custom model graph (`sentimentscope.onnx`) optimized for high-throughput deployments.

---

## 📂 Project Structure

```text
aws-project2-submission/ (Repository Root)
├── .vscode/
│   └── settings.json           # VS Code import path resolutions
├── SentimentScope/             # Project root package
│   ├── data/                   # Cache directory for datasets
│   ├── models/                 # Model checkpoints, ONNX graphs, and configs
│   │   ├── best_sentimentscope_custom.pt # Trained DemoGPT weights
│   │   ├── model_config_custom.json      # Model hyperparameters
│   │   ├── sentimentscope.onnx           # Compiled ONNX model
│   │   └── sentimentscope.onnx.data      # ONNX weights data
│   ├── results/                # Visual charts and text evaluation reports
│   │   ├── class_distribution.png
│   │   ├── review_length_distribution.png
│   │   ├── top_common_words.png
│   │   ├── wordclouds.png
│   │   ├── training_curves_custom.png
│   │   ├── confusion_matrix_custom.png
│   │   ├── eda_statistics.txt
│   │   └── evaluation_metrics_custom.txt
│   ├── src/                    # Source package
│   │   ├── __init__.py         # Package initializer
│   │   ├── dataset.py          # PyTorch custom Dataset and DataLoaders
│   │   ├── evaluate.py         # Metrics calculation and plotting
│   │   ├── inference.py        # Batch inference interface wrapper
│   │   ├── model.py            # DemoGPT and DistilBERT architectures
│   │   ├── preprocessing.py    # Text cleaning and BERT subword tokenizer wrapper
│   │   ├── train.py            # Epoch training, validation, and early stopping
│   │   └── utils.py            # Random seeds, directory setup, and EDA utilities
│   ├── main.py                 # Core CLI entrypoint orchestrator
│   └── README.md               # Package-specific documentation
├── requirements.txt            # Project dependencies
└── README.md                   # Main repository documentation
```

---

## ⚙️ How the Project was Built (Step-by-Step)

The development followed a modular machine learning engineering workflow:

1.  **Data Loading and Partitioning**: We downloaded the standard IMDB dataset programmatically using the Hugging Face `datasets` library, combining the default splits and partitioning them into **70% Training**, **15% Validation**, and **15% Testing** sets (using stratified splits to maintain class balance).
2.  **Exploratory Data Analysis (EDA)**: Rendered statistical visualizations to inspect review lengths, positive vs. negative review distribution, word frequencies (excluding stopwords), and positive/negative word clouds to understand semantic density.
3.  **Tokenization**: Selected the pre-trained `bert-base-uncased` subword tokenizer. This uses WordPiece tokenization to break down unknown words into common subword prefixes/suffixes, reducing vocabulary out-of-bounds errors.
4.  **Dataset Pipeline**: Implemented a PyTorch `Dataset` that cleans HTML tags and multiple spaces, applies subword tokenization, pads reviews to a fixed length (default `256`), and outputs token IDs alongside attention masks.
5.  **Model Customization (`DemoGPT`)**: Built the encoder layers from scratch:
    *   **Positional Encoding**: Sinusoidal waves are added to embeddings to inject token sequence order.
    *   **Multi-Head Attention**: Splits token representations into multiple heads, performs scaled dot-product attention, and applies masks to ignore padding tokens.
    *   **Pre-LayerNorm Residuals**: LayerNorm is placed *before* the attention and feed-forward sublayers for training stability.
    *   **Masked Global Average Pooling**: Calculates the average vector of non-padding tokens only, preventing padded zeroes from distorting sequence representations.
6.  **Training & Validation Loop**: Standardized an epoch-based training pipeline using the **AdamW** optimizer, a **ReduceLROnPlateau** learning rate scheduler, custom **Early Stopping** based on validation loss, and checkpoint saving.
7.  **Evaluation and ONNX Trace**: Evaluated metrics (Accuracy, Precision, Recall, F1) on the test set, rendered heatmaps, and traced the model graph with dummy inputs to serialize it to ONNX.

---

## ⚠️ Challenges Faced & Solutions

During the end-to-end implementation, we resolved several system and library challenges:

### 1. Strict Hugging Face Dataset URI Validation
*   **Challenge**: When calling `datasets.load_dataset("imdb")`, the process crashed with `huggingface_hub.errors.HfUriError: Repository id must be 'namespace/name', got 'imdb'`. Recent versions of `huggingface_hub` enforce strict namespace validations.
*   **Solution**: Modified the data loader in [`src/utils.py`](file:///d:/Aws_second_project/SentimentScope/src/utils.py) to fetch `stanfordnlp/imdb`, the official namespace-formatted version of the Large Movie Review Dataset.

### 2. PyTorch `ReduceLROnPlateau` API Changes
*   **Challenge**: Setting up the learning rate scheduler crashed with `TypeError: ReduceLROnPlateau.__init__() got an unexpected keyword argument 'verbose'`. In newer versions of PyTorch (PyTorch 2.2+), the `verbose` argument has been completely removed.
*   **Solution**: Edited the scheduler initialization in [`src/train.py`](file:///d:/Aws_second_project/SentimentScope/src/train.py) to remove the `verbose=True` parameter.

### 3. Windows Terminal Unicode Code Page 1252 Crash
*   **Challenge**: Tracing and compiling the custom model to ONNX crashed during terminal printing with `UnicodeEncodeError: 'charmap' codec can't encode character '\u2705'`. The default Windows command shell (Code Page 1252) cannot print the green checkmark emoji (✅) that PyTorch's exporter outputted in verbose mode.
*   **Solution**: Force-enabled UTF-8 terminal encoding by launching the ONNX compilation command using the environment override `$env:PYTHONIOENCODING="utf-8"`.

### 4. VS Code Static Path Import Resolution Warnings
*   **Challenge**: The VS Code editor flagged import lines in `main.py` (like `from src.utils import ...`) with red warning indicators. Because the workspace root opened in the editor was `aws-project2-submission/` rather than the subfolder `SentimentScope/`, Pylance could not resolve the paths.
*   **Solution**: 
    1. Added an empty [`__init__.py`](file:///d:/Aws_second_project/SentimentScope/src/__init__.py) in `src/` to declare it a package.
    2. Created a [`.vscode/settings.json`](file:///d:/Aws_second_project/.vscode/settings.json) configuration mapping `"python.analysis.extraPaths": ["./SentimentScope"]` so that the editor resolves subfolder packages successfully.

---

## 📈 Final Model Metrics (DemoGPT)

Trained on a subset of 15,000 reviews for 4 epochs:
*   **Test Accuracy**: **80.09%**
*   **F1-Score**: **80.87%**
*   **Precision**: **77.24%**
*   **Recall**: **84.86%**

*All visual training curves, EDA figures, and confusion matrices are saved under [`SentimentScope/results/`](file:///d:/Aws_second_project/SentimentScope/results/).*

---

## 📚 References

1.  **IMDB Dataset**: [Large Movie Review Dataset (Maas et al., 2011)](https://ai.stanford.edu/~amaas/data/sentiment/)
2.  **Attention Is All You Need**: [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762) (Original Transformer paper introducing multi-head self-attention and positional encoding).
3.  **BERT Tokenization**: [BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)](https://arxiv.org/abs/1810.04805) (Subword WordPiece tokenization strategy).
4.  **ONNX Specification**: [ONNX Runtime documentation](https://onnxruntime.ai/docs/)
5.  **Pre-LN Transformer Layer Structure**: [On Layer Normalization in the Transformer Architecture (Xiong et al., 2020)](https://arxiv.org/abs/2002.04745)
