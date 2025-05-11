# Neural Machine Translation: English-Hindi Transformer

This project implements a Neural Machine Translation (NMT) model using the Transformer architecture, specifically designed and configured for English to Hindi translation. 

## Features

-   **Transformer Architecture**: Utilizes the standard encoder-decoder Transformer model.
-   **Positional Encoding**: Supports both classic Additive Positional Encoding and advanced Rotary Positional Embedding (RoPE).
-   **Customizable Model**: Easily configure model hyperparameters like the number of layers, heads, hidden dimensions (`d_model`), and feed-forward dimensions (`d_ff`) via `config.py`.
-   **Beam Search Decoding**: Employs beam search for generating high-quality translations during inference.
-   **Efficient Training**:
    -   Mixed Precision Training for faster computation and reduced memory usage.
    -   Gradient Checkpointing to train larger models with limited GPU memory.
    -   Gradient Accumulation to simulate larger batch sizes.
    -   Learning Rate Scheduling with configurable warmup steps.
    -   Early Stopping based on validation BLEU score to prevent overfitting.
-   **Experiment Tracking**: Integrated with Weights & Biases (WandB) for logging metrics, configurations, and model performance.
-   **Dynamic Tokenization**: Builds Byte Pair Encoding (BPE) tokenizers from your data or loads existing ones.
-   **Flexible Data Handling**:
    -   Load datasets directly from Hugging Face Datasets Hub (e.g., `cfilt/iitb-english-hindi`).
    -   Use local datasets.
    -   Dataset sampling and sequence length filtering for efficient processing.
-   **Attention Visualization**: Capability to extract and potentially visualize attention scores during translation (greedy decoding path).
-   **Flash Attention Support**: Automatically utilizes `torch.nn.functional.scaled_dot_product_attention` (Flash Attention) if available in the PyTorch environment for optimized attention computation.




## Model Architecture

The model is an implementation of the standard Transformer architecture ("Attention Is All You Need") with the following key components:

-   **Input Embeddings**: Converts source and target token IDs into dense vectors.
-   **Positional Encoding**:
    -   **Additive**: Sine/cosine positional encodings added to token embeddings.
    -   **Rotary Positional Embedding (RoPE)**: Applied directly within the attention mechanism for relative position information.
-   **Encoder**:
    -   A stack of `num_encoder_layers`.
    -   Each layer consists of:
        -   Multi-Head Self-Attention (with optional RoPE).
        -   Feed-Forward Network.
    -   Layer Normalization (Pre-LN: applied before each sub-layer) and Residual Connections.
-   **Decoder**:
    -   A stack of `num_decoder_layers`.
    -   Each layer consists of:
        -   Masked Multi-Head Self-Attention (with optional RoPE).
        -   Multi-Head Cross-Attention (attends to encoder output).
        -   Feed-Forward Network.
    -   Layer Normalization (Pre-LN) and Residual Connections.
-   **Projection Layer**: A final linear layer followed by a softmax (implicitly during loss computation or explicitly for generation) to produce probability distributions over the target vocabulary.


