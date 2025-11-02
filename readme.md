# 🦙 LLaMA Architecture Study  
A clean, modular TensorFlow implementation of the **LLaMA (Decoder-only Transformer)** architecture for text summarization.

---

## 🚀 Overview

This project is a **from-scratch educational implementation** of the LLaMA architecture (Meta AI), focused on readability and modular design.

It includes:

- ✅ Token & Positional Embeddings
- ✅ Multi-Head Self-Attention with **RoPE (Rotary Positional Encoding)**
- ✅ **RMSNorm** (Root Mean Square Normalization)
- ✅ **SwiGLU** Feed-Forward activation
- ✅ Full **Transformer Decoder** architecture

> The goal is to make LLaMA internals easy to understand and modify for experimentation.

---

## 📦 Installation

```bash
git clone https://github.com/N-SriKrishna/llama-architecture-study.git
cd llama-architecture-study
pip install -r requirements.txt
```

## Quick Start
```
python scripts/train.py
```

# Generate text
```
python scripts/generate.py
```

## Project Structure
```
llama-architecture-study/
├── src/                      # Core architecture and components
│   ├── layers/              # RMSNorm, SwiGLU, RoPE, Attention
│   ├── model.py             # LLaMA decoder model
│   └── utils/               # Helpers (tokenizer, config loader, etc.)
├── scripts/
│   ├── train.py             # Training entry point
│   └── generate.py          # Inference / text generation
├── configs/                 # Model & training config files (JSON)
├── notebooks/               # Jupyter notebook version of the project
└── README.md
```

## References

LLaMA Paper: https://arxiv.org/abs/2302.13971
Attention Is All You Need: https://arxiv.org/abs/1706.03762