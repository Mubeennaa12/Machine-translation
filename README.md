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

## Overview

This project evaluates multiple pretrained multilingual translation models on English → Tamil translation using the **IndicMTEval** benchmark dataset. The best-performing model is deployed as a user-friendly Gradio web application.

**Key findings:** `facebook/nllb-200-distilled-600M` outperforms both `facebook/m2m100_418M` and `t5-base` across all automatic evaluation metrics for Tamil translation.

---

## Dataset

| Property | Value |
|---|---|
| Source | [ai4bharat/IndicMTEval](https://huggingface.co/datasets/ai4bharat/IndicMTEval) |
| Language pair | English → Tamil |
| Split used | `test` |
| Tamil samples | ~200 filtered samples |
| Fields | `src` (English), `ref` (Tamil reference), `mqm_norm_score`, `da_norm_score` |

**Preprocessing applied:**
- Lowercasing
- Removing extra whitespace
- Stripping leading/trailing spaces

---

## Models Evaluated

| Model | Parameters | Architecture | Tamil Token |
|---|---|---|---|
| `facebook/nllb-200-distilled-600M` | 600M | Encoder-Decoder (NLLB) | `tam_Taml` |
| `facebook/m2m100_418M` | 418M | Encoder-Decoder (M2M100) | `ta` |
| `t5-base` | 220M | Encoder-Decoder (T5) | prompt-based |

---

## Evaluation Results

Metrics computed on Tamil subset of IndicMTEval:

| Model | BLEU ↑ | chrF ↑ | BERTScore F1 ↑ | Cosine Sim ↑ |
|---|---|---|---|---|
| **NLLB-200 (600M)** 🏆 | **0.142** | **41.3** | **0.618** | **0.731** |
| M2M100 (418M) | 0.098 | 34.7 | 0.581 | 0.694 |
| T5-Base | 0.011 | 12.4 | 0.401 | 0.512 |

> **NLLB-200** is the clear winner — it was specifically trained with dedicated Tamil script (`tam_Taml`) support and over 200 languages, making it the best choice for low-resource Indic language translation.

### Metric Descriptions

- **BLEU**: Precision-based n-gram overlap between hypothesis and reference (0–1, higher is better)
- **chrF**: Character-level F-score — especially suited for morphologically rich languages like Tamil
- **BERTScore F1**: Contextual embedding similarity using `bert-base-multilingual-cased`
- **Cosine Similarity**: Sentence-level semantic similarity using `all-MiniLM-L6-v2`

---

## Project Structure

```
en-tamil-mt/
├── app/
│   └── app.py                  # Gradio web application
├── evaluation/
│   └── evaluate_models.py      # Full evaluation pipeline (all 3 models)
├── notebooks/
│   └── evaluation.ipynb        # Original Kaggle notebook
├── docs/
│   └── report.md               # Detailed project report
├── requirements.txt            # Python dependencies
├── .gitignore
└── README.md
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/en-tamil-mt.git
cd en-tamil-mt
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the App

```bash
python app/app.py
```

Then open your browser at `http://localhost:7860`

**Features:**
- Real-time English → Tamil translation
- Adjustable beam width and max token length
- Built-in example sentences
- Translation speed and model info display
- Dark editorial UI themed around Tamil script aesthetics

---

## Running Evaluation

Evaluate all three models:
```bash
python evaluation/evaluate_models.py --model all --samples 200
```

Evaluate only NLLB (fastest):
```bash
python evaluation/evaluate_models.py --model nllb --samples 200
```

---

## Deployment Architecture

```
User Input (English Text)
        │
        ▼
  Gradio Frontend
  (app/app.py)
        │
        ▼
  Preprocessing
  (lowercase + whitespace cleanup)
        │
        ▼
  NllbTokenizer
  (facebook/nllb-200-distilled-600M)
        │
        ▼
  NLLB Model.generate()
  forced_bos_token_id = tam_Taml
        │
        ▼
  Decoded Tamil Output
        │
        ▼
  Gradio Output Display
```

**Hardware:** Runs on CPU or GPU (auto-detected). GPU strongly recommended for batch evaluation; the Gradio app works fine on CPU for single sentences.

---

## Report

See [`docs/report.md`](docs/report.md) for the full project report including:
- Motivation and background
- Dataset analysis
- Model selection rationale
- Detailed metric analysis
- Deployment design decisions
- Limitations and future work
