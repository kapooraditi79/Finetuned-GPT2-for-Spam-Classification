# 🚀 Large Language Model for Spam Classification: Built From Scratch

A complete implementation of a GPT-based Large Language Model for SMS spam classification, built entirely from scratch using PyTorch. This project follows Sebastian Raschka's comprehensive approach in "Build a Large Language Model (From Scratch)", progressing systematically from fundamental tokenization concepts to a fully functional transformer model capable of accurate spam detection.

## 🎯 Project Overview

This implementation demonstrates the complete journey of building a production-ready language model, starting with from-scratch pretraining and transitioning to fine-tuning on OpenAI's GPT-2 pretrained weights for enhanced performance. The project religiously follows Raschka's methodical approach, ensuring both educational depth and practical applicability.

### Key Achievements
- **94% Classification Accuracy** on SMS Spam Collection dataset
- **Complete From-Scratch GPT Implementation** with all transformer components
- **Two-Stage Training Pipeline**: Custom pretraining → GPT-2 fine-tuning
- **Production-Ready Architecture** with modular, extensible design
- **Educational Implementation** following Raschka's pedagogical structure

## 🏗️ Architecture & Technical Details

### Model Configuration
```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,        # OpenAI GPT-2 vocabulary
    "context_length": 256,      # Optimized for computational efficiency
    "emb_dim": 768,            # Embedding dimension
    "n_heads": 12,             # Multi-head attention heads
    "n_layers": 12,            # Transformer layers
    "drop_rate": 0.1,          # Dropout rate for regularization
    "qkv_bias": False          # Query-Key-Value bias configuration
}
```

### Training Strategy Evolution
**Phase 1 (Chapters 1-5)**: Complete from-scratch implementation and training
**Phase 2 (Chapter 6)**: Fine-tuning on OpenAI GPT-2 pretrained weights for enhanced performance and computational efficiency

*Note: While initial training from scratch provides invaluable learning experience, leveraging pretrained GPT-2 weights acknowledges the computational reality that individual researchers cannot match billion-dollar training budgets while still achieving state-of-the-art results.*

## 📊 Dataset & Performance

### Dataset: SMS Spam Collection
- **Total Messages**: 5,574 SMS messages
- **Spam Messages**: 747 (13.4%)
- **Ham Messages**: 4,827 (86.6%)
- **Language**: English
- **Source**: UCI Machine Learning Repository

### Performance Metrics
| Metric | Score |
|--------|-------|
| **Accuracy** | 94.0% |
| **Training Approach** | From-scratch → GPT-2 fine-tuning |
| **Convergence** | Stable and efficient |
| **Architecture** | Complete transformer implementation |

## 🗂️ Project Structure & Chapter Accomplishments

### Chapter 02: Working with Text Data - Tokenization & Embeddings
**Files**: `ch2-WorkingwithTextData.ipynb`, `.ipynb_checkpoints/`, `the-verdict.txt`

**Technical Accomplishments**:
- **Advanced Tokenization Pipeline**: Implemented `GPTDatasetV1` for efficient text-to-token conversion with proper sequence handling
- **Custom DataLoader Implementation**: Built `create_dataloader_v1` with batching, shuffling, and padding strategies optimized for transformer training
- **Byte-Pair Encoding (BPE)**: Deep implementation of subword tokenization compatible with GPT-2's vocabulary
- **Text Preprocessing Pipeline**: Comprehensive data cleaning, normalization, and sequence preparation
- **Embedding Layer Foundation**: Token-to-vector conversion with learned positional encodings

*Key Innovation*: Custom dataset class handling variable-length sequences with efficient memory management for large-scale text processing.

### Chapter 03: Working with Attention Mechanisms
**Files**: `codingattentionmechanism.ipynb`

**Technical Accomplishments**:
- **Multi-Scale Attention Implementation**: Progressive development from `SelfAttention_v1` to advanced `MultiHeadAttention`
- **Causal Attention Masking**: `CausalAttention` implementation ensuring autoregressive property for language modeling
- **Attention Variants**: Multiple implementations including `SelfAttention_v2` with optimized matrix operations
- **Multi-Head Architecture**: `MultiHeadAttentionWrapper` and `MultiHeadAttention` with parallel attention heads
- **PyTorch Optimization**: `PyTorchMultiHeadAttention` leveraging `scaled_dot_product_attention` for enhanced performance
- **Attention Visualization**: Deep analysis of attention patterns and head specialization

*Key Innovation*: Comprehensive attention mechanism suite with educational progression from basic to optimized implementations.

### Chapter 04: Building GPT Model from Scratch
**Files**: `implementingGPTmodelFromScratch.ipynb`

**Technical Accomplishments**:
- **Complete Transformer Block**: `TransformerBlock` implementation with pre-norm architecture and residual connections
- **Advanced Activation Functions**: Custom `GELU` implementation for improved gradient flow and training stability
- **Layer Normalization**: `LayerNorm` with learnable parameters for training stabilization
- **Feed-Forward Networks**: `FeedForward` with configurable dimensions and dropout for regularization
- **Full GPT Architecture**: `GPTModel` assembling all components into a cohesive language model
- **Text Generation**: `generate_text_simple` for autoregressive text generation with temperature control
- **Performance Optimization**: `GPTModelFast` variant utilizing PyTorch's optimized attention operations

*Key Innovation*: Modular architecture design enabling easy experimentation with different transformer configurations and components.

### Chapter 05: Pretraining GPT Model 
**Files**: `PretrainingOnUnlabelledData.ipynb`, `gpt_download.py`, `loss-plot.pdf`, `model.pth`, `model_and_optimizer.pth`, `gpt2/124M/`

**Technical Accomplishments**:
- **Complete Training Pipeline**: `train_model_simple` with gradient accumulation, learning rate scheduling, and checkpointing
- **Advanced Text Generation**: `generate` function with nucleus sampling, temperature control, and top-k filtering
- **Comprehensive Evaluation**: `evaluate_model` with perplexity calculation and loss monitoring across validation sets
- **Model Persistence**: Robust saving/loading mechanisms with `assign` and `load_weights_into_gpt` for transfer learning
- **Tokenization Integration**: `text_to_token_ids` and `token_ids_to_text` for seamless text-model interface
- **Loss Analytics**: `calc_loss_batch`, `calc_loss_loader`, and `plot_losses` for training monitoring and debugging
- **GPT-2 Integration**: `download_and_load_gpt2` for loading OpenAI's pretrained weights and vocabulary
- **Training Visualization**: Comprehensive loss plotting and convergence analysis

*Key Innovation*: End-to-end pretraining system with transition capability to leverage pretrained GPT-2 weights, bridging educational implementation with practical performance.

### Chapter 06: Spam Classification using Finetuned GPT2
**Files**: `FinetuningForClassification.ipynb`, `accuracy-plot.pdf`, `gpt_download.py`, `loss-plot.pdf`, `previous_chapters.py`, `review_classifier.pth`, `test.csv`, `the-verdict.txt`, `train.csv`, `validation.csv`, `sms_spam_collection/`, `gpt2/124M/`

**Technical Accomplishments**:
- **Spam Dataset Pipeline**: `download_and_unzip_spam_data` with automated data acquisition and preprocessing
- **Balanced Dataset Creation**: `create_balanced_dataset` addressing class imbalance in spam detection
- **Advanced Data Splitting**: `random_split` with stratification ensuring representative train/validation/test splits
- **Custom Dataset Class**: `SpamDataset` optimized for classification with proper label encoding and sequence handling
- **Classification Metrics**: `calc_accuracy_loader` and comprehensive `evaluate_model` for multi-metric assessment
- **Fine-tuning Pipeline**: `train_classifier_simple` with classification head adaptation and transfer learning
- **Results Visualization**: `plot_values` for training curves and performance analysis
- **Production Inference**: `classify_review` for real-time spam classification with confidence scoring
- **Model Persistence**: Complete model saving/loading for deployment readiness

*Key Innovation*: Sophisticated transfer learning approach combining language model pretraining with task-specific fine-tuning, achieving 94% accuracy on spam classification.

## 🔬 Implementation Highlights

### From-Scratch Foundation (Chapters 1-5)
The initial implementation provides deep understanding of transformer internals:
- **Custom Training Loop**: Built comprehensive training infrastructure with gradient clipping, learning rate scheduling
- **Memory Management**: Efficient batch processing and sequence packing for optimal GPU utilization
- **Educational Value**: Complete transparency in model mechanics from attention computation to loss calculation

### Production Optimization (Chapter 6)
Transition to GPT-2 weights acknowledges practical constraints while maintaining technical rigor:
- **Transfer Learning**: Leveraged OpenAI's billion-parameter training while adapting for classification
- **Computational Efficiency**: Balanced educational depth with practical performance requirements
- **Real-world Applicability**: Production-ready spam classifier with deployment-ready inference

### Advanced Engineering Features
- **Modular Design**: Each component independently testable and replaceable
- **Performance Monitoring**: Comprehensive logging and visualization throughout training
- **Reproducibility**: Seed management and deterministic training procedures
- **Scalability**: Architecture supports easy scaling to larger models and datasets

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
PyTorch 2.0+
NumPy
Pandas
Matplotlib
Requests (for GPT-2 model download)
```

### Installation & Setup
```bash
git clone https://github.com/yourusername/llm-spam-classification
cd llm-spam-classification
pip install -r requirements.txt

# Download required datasets and models (handled automatically in notebooks)
```

### Quick Classification Demo
```python
# Using the trained classifier from Chapter 6
from ch06.FinetuningForClassification import classify_review

# Load the fine-tuned model
model_path = "ch06/review_classifier.pth"

# Classify messages
test_messages = [
    "Congratulations! You've won $1000. Call now!",
    "Hey, are we still meeting for lunch tomorrow?",
    "URGENT: Click here to claim your prize immediately!"
]

for message in test_messages:
    result = classify_review(message, model_path)
    print(f"Message: {message}")
    print(f"Classification: {'SPAM' if result > 0.5 else 'HAM'} (confidence: {result:.3f})")
```

## 📈 Technical Deep Dive

### Attention Mechanism Evolution
The implementation showcases attention development from basic to optimized:
- **Self-Attention v1**: Basic implementation for educational clarity
- **Self-Attention v2**: Optimized matrix operations and memory efficiency
- **Causal Attention**: Autoregressive masking for language modeling
- **Multi-Head Attention**: Parallel attention computation with head concatenation
- **PyTorch Optimized**: Leveraging `scaled_dot_product_attention` for production speed

### Training Pipeline Architecture
**Two-Stage Approach**:
1. **Educational Phase**: Complete from-scratch implementation demonstrating every component
2. **Practical Phase**: GPT-2 fine-tuning acknowledging computational realities while maintaining technical rigor

### Model Architecture Decisions
- **Context Length (256)**: Optimized for SMS message characteristics and computational efficiency
- **12 Layers/12 Heads**: Balanced model complexity suitable for classification tasks
- **Embedding Dimension (768)**: Sufficient representational capacity without over-parameterization
- **Dropout (0.1)**: Regularization preventing overfitting on relatively small spam dataset

## 🎓 Learning Outcomes & Skills Demonstrated

### Technical Expertise
- **Deep Learning Fundamentals**: Neural network architecture design and optimization
- **Transformer Architecture**: Complete understanding of attention mechanisms and language modeling
- **PyTorch Mastery**: Advanced model implementation, training loops, and optimization
- **NLP Pipeline**: End-to-end text processing from tokenization to classification
- **Transfer Learning**: Practical application of pretrained models for domain-specific tasks

### Software Engineering
- **Modular Design**: Clean, maintainable code structure following software engineering principles
- **Documentation**: Comprehensive code documentation and educational explanations
- **Version Control**: Systematic development following the book's chapter structure
- **Performance Optimization**: Memory-efficient implementations and computational optimizations

### Research & Development
- **Systematic Approach**: Following Raschka's methodical development methodology
- **Experimental Design**: Proper train/validation/test splits and evaluation metrics
- **Results Analysis**: Comprehensive performance evaluation and visualization
- **Practical Application**: Real-world spam classification with production-ready inference

## 🛠️ Framework & Dependencies

**Core Implementation**: Pure PyTorch without high-level frameworks
- Demonstrates deep understanding of transformer internals
- Full control over model architecture and training dynamics
- Educational value for understanding LLM fundamentals
- Production-ready performance through optimization

## 📚 References & Acknowledgments

This implementation is based on Sebastian Raschka's exceptional work:
- **Primary Source**: Raschka, Sebastian. "Build a Large Language Model (From Scratch)"
- **Methodology**: Faithful implementation following the book's systematic approach
- **Educational Framework**: Raschka's pedagogical structure from basics to advanced concepts

### Additional References
- Vaswani et al. "Attention Is All You Need" (2017)
- Radford et al. "Language Models are Unsupervised Multitask Learners" (2019)
- OpenAI GPT-2 Model and Weights
- SMS Spam Collection Dataset (UCI ML Repository)

## 🤝 Contributing

This educational implementation demonstrates the complete LLM development pipeline. For improvements:
1. Fork the repository
2. Follow the chapter-based structure
3. Maintain educational clarity while optimizing performance
4. Submit pull requests with detailed technical explanations


---

*Built following Sebastian Raschka's "Build a Large Language Model (From Scratch)" - a comprehensive journey from tokenization fundamentals to production-ready spam classification. This implementation showcases both the educational depth of from-scratch development and the practical wisdom of leveraging pretrained models for optimal results.*

**Special Thanks**: Sebastian Raschka for providing the definitive guide to understanding and implementing Large Language Models from first principles.
