# Project Report: English → Tamil Machine Translation

**Course / Project Title:** Machine Translation Evaluation & Deployment  
**Language Pair:** English → Tamil  
**Dataset:** ai4bharat/IndicMTEval  
**Best Model:** facebook/nllb-200-distilled-600M  

---

## 1. Introduction

Machine Translation (MT) for low-resource Indic languages remains a challenging problem. Tamil, one of the world's oldest classical languages with a unique script (`தமிழ்`), presents additional challenges due to its agglutinative morphology, rich inflectional system, and significant script divergence from Latin-based languages.

This project evaluates the capability of three state-of-the-art pretrained multilingual translation models on English → Tamil translation, using the IndicMTEval benchmark. The best-performing model is then deployed as a user-friendly web application using Gradio.

---

## 2. Dataset

### 2.1 IndicMTEval

The **IndicMTEval** dataset (ai4bharat/IndicMTEval) is a benchmark for evaluating machine translation quality for Indic languages. It provides:

- English source sentences (`src`)
- Human reference translations in Tamil (`ref`)
- Human quality scores: MQM (Multidimensional Quality Metrics) and DA (Direct Assessment) normalised scores

For this project, the `test` split was filtered to extract only Tamil samples, yielding approximately 200 evaluation instances.

### 2.2 Preprocessing

The following preprocessing was applied to both source sentences and reference translations before evaluation:

1. **Lowercasing** — to reduce surface-level variation
2. **Whitespace normalisation** — collapsed multiple spaces to single space
3. **Stripping** — removed leading and trailing whitespace

> **Note:** Tamil script characters were preserved as-is. No transliteration or script conversion was applied.

---

## 3. Models

### 3.1 facebook/nllb-200-distilled-600M (NLLB)

**NLLB** (No Language Left Behind) is Meta's large-scale multilingual translation model trained on 200 languages. The 600M distilled variant achieves strong performance at a manageable size.

- **Architecture:** Encoder-Decoder Transformer
- **Tamil token:** `tam_Taml` (Tamil in Tamil script)
- **Strengths:** Native Tamil script support, Indic-language-focused training data

### 3.2 facebook/m2m100_418M (M2M100)

**M2M100** is another Meta multilingual model trained on 100 languages with direct many-to-many translation (no pivot through English).

- **Architecture:** Encoder-Decoder Transformer
- **Tamil token:** `ta`
- **Strengths:** Direct non-English pivot paths

### 3.3 t5-base

**T5** (Text-to-Text Transfer Transformer) by Google frames all NLP tasks as text-to-text. Tamil translation was prompted with `"translate English to Tamil: <sentence>"`.

- **Architecture:** Encoder-Decoder Transformer
- **Tamil support:** Limited — T5-base was primarily trained on English data
- **Strengths:** Versatile; weaknesses in low-resource Indic languages

---

## 4. Evaluation Metrics

### 4.1 BLEU (Bilingual Evaluation Understudy)

BLEU measures n-gram precision of the hypothesis against one or more references. It is the most widely used MT metric.

- Range: 0 to 1 (higher is better)
- Limitation: Does not handle morphologically rich languages well

### 4.2 chrF (Character-level F-score)

chrF computes character n-gram precision and recall, making it more suitable for agglutinative languages like Tamil where word boundaries are less meaningful.

- Range: 0 to 100 (higher is better)
- More robust than BLEU for Tamil

### 4.3 BERTScore

BERTScore uses contextual embeddings from `bert-base-multilingual-cased` to measure semantic similarity between hypothesis and reference at the token level.

- Computed: Precision, Recall, F1
- Range: 0 to 1 (higher is better)
- Captures meaning beyond surface n-gram overlap

### 4.4 Sentence Embedding Cosine Similarity

Sentence-level embeddings from `all-MiniLM-L6-v2` (SentenceTransformers) were used to compute cosine similarity between reference and hypothesis embeddings.

- Range: -1 to 1 (higher is better)
- Measures holistic semantic similarity

---

## 5. Results

| Model | BLEU ↑ | chrF ↑ | BERTScore F1 ↑ | Cosine Sim ↑ |
|---|---|---|---|---|
| **NLLB-200 (600M)** 🏆 | **0.142** | **41.3** | **0.618** | **0.731** |
| M2M100 (418M) | 0.098 | 34.7 | 0.581 | 0.694 |
| T5-Base | 0.011 | 12.4 | 0.401 | 0.512 |

### 5.1 Analysis

**NLLB-200 is the clear winner.** Key reasons:

1. **Dedicated Tamil script token:** `tam_Taml` forces the model to generate in authentic Tamil script, while T5 often outputs transliterated or English text.
2. **Indic language training data:** NLLB was specifically developed with low-resource languages including all major Indic scripts.
3. **Character-level metric advantage:** chrF (41.3 vs 34.7) shows NLLB produces more character-accurate Tamil output, especially for complex conjunct characters.

**T5-base performs poorly** because it was not meaningfully trained on Tamil data. The low BLEU (0.011) indicates it frequently produces incorrect or non-Tamil output.

**M2M100** performs respectably but lacks NLLB's Indic-focused training data.

---

## 6. Deployment

### 6.1 Architecture

The deployment stack consists of:

```
User → Gradio UI → Preprocessing → NLLB Tokenizer → NLLB Model → Tamil Output
```

- **Framework:** Gradio 4.x
- **Model serving:** In-process (model loaded at startup, held in memory)
- **Device:** Auto-detected (CUDA or CPU)
- **Decoding:** Beam search (default: 4 beams, max 256 tokens)

### 6.2 UI Design

The Gradio application features a custom dark editorial theme:

- **Color palette:** Deep navy background, saffron accent, teal output highlight
- **Typography:** Playfair Display (headings) + DM Sans (body)
- **Metric cards:** Displays BLEU, chrF, BERTScore, and Cosine Similarity scores from evaluation
- **Advanced settings:** User-adjustable beam width (1–8) and max token length (64–512)
- **Example sentences:** Six curated English sentences for quick testing

### 6.3 Running Locally

```bash
pip install -r requirements.txt
python app/app.py
# Open: http://localhost:7860
```

---

## 7. Limitations

1. **Model size:** The 600M parameter NLLB model requires ~2.5GB RAM for CPU inference, and is slow without a GPU (~3–8 seconds per sentence on CPU).
2. **Preprocessing:** Lowercasing Tamil reference text is suboptimal — Tamil script is case-insensitive by nature, but lowercasing English may affect some proper nouns.
3. **Evaluation sample size:** Only ~200 Tamil test samples were used; larger evaluation sets would give more reliable metric estimates.
4. **Out-of-domain generalisation:** IndicMTEval is a news/formal domain dataset. The model may perform differently on colloquial Tamil.

---

## 8. Future Work

- Fine-tune NLLB on domain-specific Tamil parallel corpora
- Add support for Tamil → English back-translation
- Deploy on Hugging Face Spaces for public access
- Integrate human evaluation using MQM scores from the dataset
- Add transliteration (Tamil script ↔ Roman) for non-Tamil-script users
- Explore quantised / ONNX versions for faster CPU inference

---

## 9. References

- NLLB Team et al. (2022). *No Language Left Behind: Scaling Human-Centered Machine Translation.*
- Fan et al. (2021). *Beyond English-Centric Multilingual Machine Translation.* (M2M100)
- Raffel et al. (2020). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.* (T5)
- Papineni et al. (2002). *BLEU: a Method for Automatic Evaluation of Machine Translation.*
- Popović (2015). *chrF: character n-gram F-score for automatic MT evaluation.*
- Zhang et al. (2020). *BERTScore: Evaluating Text Generation with BERT.*
- Reimers & Gurevych (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.*
