
# 🤖 GPT Mastery: A Comprehensive Guide To Crafting Your Own ChatGPT
A complete from-scratch implementation of a GPT-based language model, inspired by **Andrej Karpathy’s Zero to Hero series**. This was the final project submission for the "Make Your Own ChatGPT" SOC 2024.

---

## 🧱 Project Structure

| Week | Notebook                       | Focus Area                         |
|------|--------------------------------|------------------------------------|
| 1    | `23B1234_WEEK_1.ipynb`         | Tokenizer, Dataset, Bigram Model   |
| 2    | `23B1234_WEEK_2.ipynb`         | Self-Attention, Head Implementation|
| 3    | `23B1234_WEEK_3.ipynb`         | Multi-Head Attention, MLP, Block   |
| 5    | `23B1234_WEEK_5.ipynb`         | GPT Model + Training from Scratch  |
| Final| `23B1234_Ultimate_Project.ipynb` | Full GPT + Dataset of Choice + Parallel Attention |

---

## 📚 Week-wise Overview

### 🔹 Week 1: Tokenization & Bigram Language Modeling
- Built a character-level tokenizer.
- Trained a basic Bigram model using raw logits over token space.
- Implemented a mini dataloader and sampling loop.

### 🔹 Week 2: Self-Attention Heads
- Developed a single `Head` class to implement scaled dot-product attention.
- Introduced masking for autoregressive behavior.
- Learned about causality and broadcast mechanics in PyTorch.

### 🔹 Week 3: MultiHeadAttention, MLP & Transformer Block
- Implemented multiple attention heads and combined outputs via projection.
- Added a simple MLP (2-layer FFN with GELU).
- Created a `Block` class: Attention + MLP + LayerNorm + Residuals.

### 🔹 Week 5: Assembling the GPT Model
- Assembled everything into a complete `GPT` class.
- Introduced positional embeddings, loss computation, and text generation.
- Trained a small GPT from scratch using minimal dependencies.

---

## 🧠 Final Project: GPT Mastery

### 🚀 Objective
1. **Implement parallel multi-head attention**  
   - Refactored `Head` and `MultiHeadAttention` into a unified parallelized module using tensor manipulation (like in `nanoGPT`).
   - Treated attention heads as a batch dimension for efficiency.

2. **Train GPT on a custom dataset**  
   - Chose a unique dataset (e.g., math problems, code, poetry).
   - Optionally explored digit-by-digit addition using left-to-right loss masking.
   - Used `CrossEntropyLoss(ignore_index=-1)` for masked supervision.

### ✨ Stretch Goals (Explored/Planned)
- Trained transformer to perform simple arithmetic (`a + b = c`) via generation.
- Planned future extensions: calculator GPT for `+ - * /` using Chain of Thought prompts.

---
