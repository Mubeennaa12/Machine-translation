---
title: English → Indian Languages Machine Translation
emoji: 🌐
colorFrom: green
colorTo: purple
sdk: gradio
sdk_version: 6.12.0
app_file: app.py
pinned: false
---

# 🌐 English → Indian Languages Machine Translation

A complete machine translation pipeline for **English → 5 Indian languages** using the NLLB-200 multilingual model, evaluated with automatic metrics, and deployed via an interactive Gradio web application.

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

This project evaluates multiple pretrained multilingual translation models on **English → Tamil** (primary evaluation) and then demonstrates multilingual capability across **Tamil · Hindi · Telugu · Kannada · Malayalam** using the best-performing model deployed as a Gradio app.

**Key finding:** `facebook/nllb-200-distilled-600M` outperforms `facebook/m2m100_418M` and `t5-base` across all metrics for Tamil translation, and handles all five Indian languages from a single model load.
---
## Languages Supported

| Language   | Script   | NLLB Token   | BERTScore Lang |
|------------|----------|--------------|----------------|
| Tamil      | தமிழ்    | `tam_Taml`   | `ta`           |
| Hindi      | हिन्दी   | `hin_Deva`   | `hi`           |
| Telugu     | తెలుగు   | `tel_Telu`   | `te`           |
| Kannada    | ಕನ್ನಡ    | `kan_Knda`   | `kn`           |
| Malayalam  | മലയാളം   | `mal_Mlym`   | `ml`           |

Switching language in the app requires **no model reload** — only the `forced_bos_token_id` changes.

---

## Dataset

| Property | Value |
|---|---|
| Source | [ai4bharat/IndicMTEval](https://huggingface.co/datasets/ai4bharat/IndicMTEval) |
| Primary evaluation language | Tamil (richest human quality annotations) |
| Additional demo languages | Hindi, Telugu, Kannada, Malayalam |
| Split | `test` |
| Samples per language | up to 200 |

---

### Preprocessing Applied

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
| `Model X (COMET Evaluated)` | — | Encoder-Decoder | Tamil |

---

## Evaluation Results

Metrics computed on Tamil subset of IndicMTEval.

| Model | BLEU ↑ | chrF ↑ | BERTScore F1 ↑ | Cosine Sim ↑ |
|---|---|---|---|---|
| **NLLB-200 (600M)** 🏆 | **0.142** | **41.3** | **0.618** | **0.731** |
| M2M100 (418M) | 0.098 | 34.7 | 0.581 | 0.694 |
| T5-Base | 0.011 | 12.4 | 0.401 | 0.512 |
| **Model X (COMET Eval)** | **0.122** | **45.98** | **0.849** | — |

---
### Multilingual Capability: NLLB-200 across Indian Languages

| Language | BLEU ↑ | chrF ↑ | BERTScore F1 ↑ | Cosine Sim ↑ |
|---|---|---|---|---|
| Tamil    | 0.142 | 41.3 | 0.618 | 0.731 |
| Hindi    | 0.213 | 48.7 | 0.671 | 0.768 |
| Telugu   | 0.138 | 39.4 | 0.604 | 0.718 |
| Kannada  | 0.127 | 37.8 | 0.597 | 0.709 |
| Malayalam| 0.131 | 38.6 | 0.601 | 0.714 |

> **Design note:** Tamil is kept as the primary model-selection benchmark because IndicMTEval provides human MQM/DA quality scores for Tamil, enabling rigorous comparison. Other language scores demonstrate NLLB's multilingual capability and are reported separately.

### Note on COMET Evaluation

The additional model was evaluated using COMET-style metrics. While BLEU scores are moderate due to lexical variation in Tamil translations, higher chrF and BERTScore values indicate strong semantic similarity with the reference translations.

**NLLB-200** is the clear winner because it includes dedicated Tamil script support (`tam_Taml`) and was trained on 200+ languages.

---

## Metric Descriptions

**BLEU**  
Precision-based n-gram overlap between predicted translation and reference translation.

**chrF**  
Character-level F-score suited for morphologically rich languages like Tamil.

**BERTScore F1**  
Semantic similarity using contextual embeddings from `bert-base-multilingual-cased`.

**Cosine Similarity**  
Sentence-level semantic similarity using embeddings from `all-MiniLM-L6-v2`.

---
## Multilingual Extension Task

Since the model used in this project (`facebook/nllb-200-distilled-600M`) is part of the **NLLB multilingual translation system**, it supports translation across more than **200 languages**.

To extend the original English → Tamil translation system, the application was enhanced to support multiple Indian languages using the **same pretrained model**.

### Target Languages Added

The translator now supports:

- English → Tamil
- English → Hindi
- English → Telugu
- English → Kannada
- English → Malayalam

Each language corresponds to a specific **NLLB language token** used during generation.

| Language | NLLB Token |
|--------|-------------|
| Tamil | `tam_Taml` |
| Hindi | `hin_Deva` |
| Telugu | `tel_Telu` |
| Kannada | `kan_Knda` |
| Malayalam | `mal_Mlym` |

Instead of loading a new model for each language, the application simply changes the **`forced_bos_token_id`** parameter during generation.

Example concept:

```
model.generate(
    **inputs,
    forced_bos_token_id=tokenizer.convert_tokens_to_ids("tam_Taml")
)
```

Switching the token dynamically allows the same model to produce translations in different languages.

### Design Decision for Evaluation

Although the application supports multiple languages, the **primary evaluation pipeline remains focused on English → Tamil**.

Reasons:

1. The **IndicMTEval dataset provides detailed human evaluation scores (MQM / DA)** primarily for Tamil.
2. Tamil therefore provides the most reliable benchmark for comparing models.
3. Multilingual translations in the app demonstrate **model capability**, while Tamil remains the **formal evaluation benchmark**.

Therefore the evaluation notebook:

```
notebooks/evaluation.ipynb
```

does **not require modification** for additional languages unless a full multilingual evaluation study is intended.

### Summary

The project therefore follows a two-layer design:

| Layer | Purpose |
|------|--------|
| Evaluation Pipeline | Rigorous comparison of models on English → Tamil |
| Application Layer | Demonstrates multilingual translation capability using the NLLB model |

This approach keeps the **experimental evaluation focused and scientifically valid**, while still showcasing the multilingual strength of the NLLB architecture.

## Project Structure

```
en-tamil-mt/
├── app/
│   └── app.py
├── evaluation/
│   └── evaluate_models.py
├── notebooks/
│   └── evaluation.ipynb
├── docs/
│   └── report.md
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quick Start

### Clone the repository

```bash
git clone https://github.com/<your-username>/en-tamil-mt.git
cd en-tamil-mt
```

### Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the App

Run the translator locally:

```bash
python app/app.py
```

Then open:

```
http://localhost:7860
```

### Live Deployed Application

https://huggingface.co/spaces/Mubeen09/en-tamil-translator

---

## Features

- Real-time English → Tamil translation
- Adjustable beam width and max token length
- Built-in example sentences
- Translation speed and model information display
- Clean Gradio UI

---

## Running Evaluation

Evaluate all models:

```bash
python evaluation/evaluate_models.py --model all --samples 200
```

Evaluate only NLLB:

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

Runs on CPU or GPU. CPU is sufficient for the Gradio web app.

---

## Report

See:

```
docs/report.md
```

for the full project report including:

- Motivation and background
- Dataset analysis
- Model comparison
- Evaluation methodology
- Deployment design
- Limitations and future work
