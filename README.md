---
title: English → Tamil Machine Translation
emoji: 🌐
colorFrom: green
colorTo: purple
sdk: gradio
sdk_version: 6.12.0
app_file: app.py
pinned: false
---

# 🌐 English → Tamil Machine Translation

A complete machine translation pipeline for **English → Tamil** using state-of-the-art pretrained multilingual models, evaluated with automatic metrics, and deployed via an interactive Gradio web application.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Models Evaluated](#models-evaluated)
- [Evaluation Results](#evaluation-results)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Running the App](#running-the-app)
- [Running Evaluation](#running-evaluation)
- [Deployment Architecture](#deployment-architecture)
- [Report](#report)

---

# Overview

This project evaluates multiple pretrained multilingual translation models on **English → Tamil** translation using the **IndicMTEval** benchmark dataset. The best-performing model is deployed as a user-friendly **Gradio web application**.

**Key findings:**  
`facebook/nllb-200-distilled-600M` outperforms both `facebook/m2m100_418M` and `t5-base` across all automatic evaluation metrics for Tamil translation.

---

# Dataset

| Property | Value |
|---|---|
| Source | https://huggingface.co/datasets/ai4bharat/IndicMTEval |
| Language pair | English → Tamil |
| Split used | `test` |
| Tamil samples | ~200 filtered samples |
| Fields | `src` (English), `ref` (Tamil reference), `mqm_norm_score`, `da_norm_score` |

### Preprocessing applied

- Lowercasing
- Removing extra whitespace
- Stripping leading/trailing spaces

---

# Models Evaluated

| Model | Parameters | Architecture | Tamil Token |
|---|---|---|---|
| `facebook/nllb-200-distilled-600M` | 600M | Encoder-Decoder (NLLB) | `tam_Taml` |
| `facebook/m2m100_418M` | 418M | Encoder-Decoder (M2M100) | `ta` |
| `t5-base` | 220M | Encoder-Decoder (T5) | prompt-based |

---

# Evaluation Results

Metrics computed on Tamil subset of IndicMTEval.

| Model | BLEU ↑ | chrF ↑ | BERTScore F1 ↑ | Cosine Sim ↑ |
|---|---|---|---|---|
| **NLLB-200 (600M)** 🏆 | **0.142** | **41.3** | **0.618** | **0.731** |
| M2M100 (418M) | 0.098 | 34.7 | 0.581 | 0.694 |
| T5-Base | 0.011 | 12.4 | 0.401 | 0.512 |

**NLLB-200** is the clear winner. It was specifically trained with dedicated Tamil script support (`tam_Taml`) and over **200 languages**, making it highly effective for low-resource Indic language translation.

---

# Metric Descriptions

**BLEU**  
Precision-based n-gram overlap between hypothesis and reference.

**chrF**  
Character-level F-score suited for morphologically rich languages like Tamil.

**BERTScore F1**  
Semantic similarity using contextual embeddings from `bert-base-multilingual-cased`.

**Cosine Similarity**  
Sentence-level semantic similarity using embeddings from `all-MiniLM-L6-v2`.

---

# Project Structure
