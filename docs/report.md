# Project Report: English → Tamil Machine Translation

**Course / Project Title:** Machine Translation Evaluation & Deployment  
**Language Pair:** English → Tamil  
**Dataset:** ai4bharat/IndicMTEval  
**Best Model:** facebook/nllb-200-distilled-600M  

---

# 1. Introduction

Machine Translation (MT) for low-resource Indic languages remains a challenging problem. Tamil, one of the world's oldest classical languages with a unique script (`தமிழ்`), presents additional challenges due to its agglutinative morphology, rich inflectional system, and significant script divergence from Latin-based languages.

This project evaluates the capability of three state-of-the-art pretrained multilingual translation models on English → Tamil translation using the IndicMTEval benchmark. The best-performing model is then deployed as a user-friendly web application using Gradio.

In addition to the primary Tamil evaluation, further experiments were conducted to analyze the **multilingual capabilities of the NLLB model across multiple Indian languages**.

---

# 2. Dataset

## 2.1 IndicMTEval

The **IndicMTEval dataset (ai4bharat/IndicMTEval)** is a benchmark dataset designed for evaluating machine translation systems for Indic languages.

It provides:

- English source sentences (`src`)
- Human reference translations (`ref`)
- Human quality scores:
  - MQM (Multidimensional Quality Metrics)
  - DA (Direct Assessment)

For this project, the **test split** was filtered to extract only Tamil samples, yielding approximately **200 evaluation instances**.

---

## 2.2 Preprocessing

The following preprocessing steps were applied before evaluation:

1. **Lowercasing** — reduces surface-level variation  
2. **Whitespace normalization** — multiple spaces replaced with single space  
3. **Trimming** — leading and trailing spaces removed  

Tamil script characters were preserved as-is. No transliteration or script conversion was applied.

---

# 3. Models

## 3.1 NLLB (facebook/nllb-200-distilled-600M)

**NLLB (No Language Left Behind)** is Meta’s multilingual translation model trained on over **200 languages**.

- Architecture: Encoder–Decoder Transformer  
- Tamil token: `tam_Taml`  
- Parameters: 600M  
- Strengths:
  - Native Tamil script support
  - Extensive multilingual training data
  - Designed specifically for low-resource languages

---

## 3.2 M2M100 (facebook/m2m100_418M)

**M2M100** is a multilingual model trained on **100 languages** capable of many-to-many translation.

- Architecture: Encoder–Decoder Transformer  
- Tamil token: `ta`  
- Strengths:
  - Direct translation between language pairs
  - No need for English pivot translation

---

## 3.3 T5-base

**T5 (Text-to-Text Transfer Transformer)** by Google treats every NLP task as a text-to-text problem.

Tamil translation was performed using prompts such as:

```
translate English to Tamil: <sentence>
```

- Architecture: Encoder–Decoder Transformer  
- Tamil support: Limited  
- Weakness: Mostly trained on English data

---

# 4. Evaluation Metrics

## 4.1 BLEU

BLEU (Bilingual Evaluation Understudy) measures **n-gram precision overlap** between predicted translation and reference.

Range: 0 → 1  
Higher values indicate better lexical match.

Limitation: performs poorly on morphologically rich languages.

---

## 4.2 chrF

chrF measures **character n-gram similarity**, making it more suitable for languages like Tamil.

Range: 0 → 100  
Better for agglutinative languages.

---

## 4.3 BERTScore

BERTScore uses contextual embeddings from:

```
bert-base-multilingual-cased
```

It computes semantic similarity between predicted and reference translations.

Range: 0 → 1

---

## 4.4 Sentence Embedding Cosine Similarity

Sentence embeddings from:

```
all-MiniLM-L6-v2
```

were used to compute cosine similarity between translations.

Range: −1 → 1

---

## 4.5 COMET (WMT20 COMET-DA)

COMET is a neural evaluation metric trained to predict human translation quality scores.

Advantages:

- correlates strongly with human evaluation
- captures semantic quality beyond surface similarity

---

# 5. Results (English → Tamil)

| Model | BLEU ↑ | chrF ↑ | BERTScore F1 ↑ | Cosine Sim ↑ |
|---|---|---|---|---|
| **NLLB-200 (600M)** 🏆 | **0.142** | **41.3** | **0.618** | **0.731** |
| M2M100 (418M) | 0.098 | 34.7 | 0.581 | 0.694 |
| T5-Base | 0.011 | 12.4 | 0.401 | 0.512 |

## Analysis

**NLLB-200 performs best across all metrics.**

Reasons:

1. Dedicated Tamil script token (`tam_Taml`)
2. Training focused on low-resource languages
3. Better character-level accuracy (chrF)

T5 performs poorly because it was not meaningfully trained on Tamil data.

---

# 6. Cross-Language Evaluation

To analyze multilingual capability, additional experiments were performed across several Indian languages using the **NLLB model in a zero-shot setting**.

Languages evaluated:

- Hindi
- Tamil
- Malayalam
- Marathi
- Gujarati

Dataset size per language:

```
800 training samples
200 validation samples
```

---

## Validation Results

| Language | BLEU ↑ | chrF ↑ | COMET ↑ | Ensemble Score ↑ |
|---|---|---|---|---|
| Hindi | 31.77 | 37.05 | 0.4404 | 47.20 |
| Tamil | 8.45 | 58.68 | **0.5964** | 49.29 |
| Malayalam | 11.95 | 60.29 | 0.5612 | 50.38 |
| Marathi | **37.04** | **78.40** | 0.4397 | **62.57** |
| Gujarati | 21.03 | 56.23 | 0.5830 | 52.41 |

---

## Best Performers

| Metric | Best Language | Score |
|---|---|---|
| BLEU | Marathi | 37.04 |
| chrF | Marathi | 78.40 |
| COMET | Tamil | **0.5964** |
| Ensemble Score | Marathi | **62.57** |

---

## Observations

Average Ensemble Score: **52.37 / 100**

Best Score: **62.57 (Marathi)**  
Worst Score: **47.20 (Hindi)**  

Spread across languages: **15.37 points**

Interpretation:

- Marathi shows strongest lexical alignment
- Tamil shows strongest semantic alignment
- NLLB provides stable multilingual translation performance

---

# 7. Deployment

## Architecture

```
User
 ↓
Gradio UI
 ↓
Preprocessing
 ↓
NLLB Tokenizer
 ↓
NLLB Model
 ↓
Target Language Token
 ↓
Translation Output
```

---

## UI Design

Features include:

- Dark theme UI
- Language selection
- Beam search adjustment
- Example sentences
- Metric display cards

---

## Running Locally

```
pip install -r requirements.txt
python app/app.py
```

Open:

```
http://localhost:7860
```

---

# 8. Limitations

1. Large model size (~600M parameters)
2. CPU inference latency
3. Small evaluation sample size (~200)
4. Domain mismatch between training and evaluation data

---

# 9. Future Work

- Fine-tuning on larger Tamil parallel corpora
- Add Tamil → English translation
- Deploy scalable API
- Incorporate human evaluation
- Add transliteration support

---

# 10. References

NLLB Team et al. (2022) — *No Language Left Behind*

Fan et al. (2021) — *M2M100: Many-to-Many Multilingual Translation*

Raffel et al. (2020) — *Exploring the Limits of Transfer Learning with T5*

Papineni et al. (2002) — *BLEU Metric*

Popović (2015) — *chrF Metric*

Zhang et al. (2020) — *BERTScore*

Reimers & Gurevych (2019) — *Sentence-BERT*