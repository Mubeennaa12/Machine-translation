"""
evaluation/evaluate_models.py
─────────────────────────────────────────────────────────────────────────────
Evaluates three pretrained MT models on the ai4bharat/IndicMTEval Tamil subset.

Models evaluated:
  1. facebook/nllb-200-distilled-600M  (NLLB)
  2. facebook/m2m100_418M              (M2M100)
  3. t5-base                           (T5)

Metrics:
  - BLEU
  - chrF
  - BERTScore F1
  - Sentence Embedding Cosine Similarity

Usage:
  python evaluate_models.py [--model nllb|m2m|t5|all] [--samples 200]
"""

import re
import argparse
import numpy as np
import evaluate
import sacrebleu
from datasets import load_dataset
from transformers import (
    NllbTokenizer, AutoModelForSeq2SeqLM,
    M2M100Tokenizer, M2M100ForConditionalGeneration,
    T5Tokenizer, T5ForConditionalGeneration,
)
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ─────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────
def load_tamil_data(num_samples: int = 200):
    print("Loading IndicMTEval dataset...")
    dataset = load_dataset("ai4bharat/IndicMTEval", split="test")
    dataset_tamil = dataset.filter(lambda x: x["language"] == "Tamil")
    if num_samples:
        dataset_tamil = dataset_tamil.select(range(min(num_samples, len(dataset_tamil))))
    sources = [preprocess(t) for t in dataset_tamil["src"]]
    references = [preprocess(t) for t in dataset_tamil["ref"]]
    print(f"Loaded {len(sources)} Tamil samples.")
    return sources, references


# ─────────────────────────────────────────
# Model Translators(nllb)
# ─────────────────────────────────────────
def translate_nllb(sources, batch_size=8):
    print("\n[NLLB] Loading model...")
    tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to(DEVICE)
    model.eval()
    preds = []
    for i in range(0, len(sources), batch_size):
        batch = sources[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids("tam_Taml"),
                num_beams=4, max_length=256,
            )
        preds.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
        print(f"  NLLB: {min(i+batch_size, len(sources))}/{len(sources)}", end="\r")
    return [preprocess(p) for p in preds]


def translate_m2m(sources, batch_size=8):
    print("\n[M2M100] Loading model...")
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M").to(DEVICE)
    model.eval()
    tokenizer.src_lang = "en"
    preds = []
    for i in range(0, len(sources), batch_size):
        batch = sources[i:i+batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        with torch.no_grad():
            out = model.generate(
                **encoded,
                forced_bos_token_id=tokenizer.get_lang_id("ta"),
                num_beams=4, max_length=256,
            )
        preds.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
        print(f"  M2M: {min(i+batch_size, len(sources))}/{len(sources)}", end="\r")
    return [preprocess(p) for p in preds]


def translate_t5(sources, batch_size=8):
    print("\n[T5] Loading model...")
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained("t5-base").to(DEVICE)
    model.eval()
    preds = []
    for i in range(0, len(sources), batch_size):
        batch = sources[i:i+batch_size]
        inputs = tokenizer(
            ["translate English to Tamil: " + s for s in batch],
            return_tensors="pt", padding=True, truncation=True,
        ).to(DEVICE)
        with torch.no_grad():
            out = model.generate(**inputs, max_length=256)
        preds.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
        print(f"  T5: {min(i+batch_size, len(sources))}/{len(sources)}", end="\r")
    return [preprocess(p) for p in preds]


# ─────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────
def evaluate_predictions(predictions, references, model_name="Model"):
    # Filter empty pairs
    pairs = [(p, r) for p, r in zip(predictions, references) if p.strip() and r.strip()]
    clean_preds, clean_refs = zip(*pairs)

    print(f"\n  Evaluating {model_name} on {len(clean_preds)} valid samples...")

    # BLEU
    bleu = evaluate.load("bleu")
    bleu_score = bleu.compute(predictions=list(clean_preds), references=[[r] for r in clean_refs])["bleu"]

    # chrF
    chrf_score = sacrebleu.corpus_chrf(list(clean_preds), [list(clean_refs)]).score

    # BERTScore
    _, _, F1 = bert_score(list(clean_preds), list(clean_refs), model_type="bert-base-multilingual-cased")
    bert_f1 = F1.mean().item()

    # Embedding Cosine Similarity
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    ref_emb = embed_model.encode(list(clean_refs))
    pred_emb = embed_model.encode(list(clean_preds))
    cos_sim = cosine_similarity(ref_emb, pred_emb).diagonal().mean()

    return {
        "model": model_name,
        "BLEU": round(bleu_score, 4),
        "chrF": round(chrf_score, 2),
        "BERTScore_F1": round(bert_f1, 4),
        "EmbeddingSim": round(float(cos_sim), 4),
    }


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all", choices=["nllb", "m2m", "t5", "all"])
    parser.add_argument("--samples", type=int, default=200)
    args = parser.parse_args()

    sources, references = load_tamil_data(args.samples)

    results = []

    if args.model in ("nllb", "all"):
        preds = translate_nllb(sources)
        results.append(evaluate_predictions(preds, references, "NLLB-200 (600M)"))

    if args.model in ("m2m", "all"):
        preds = translate_m2m(sources)
        results.append(evaluate_predictions(preds, references, "M2M100 (418M)"))

    if args.model in ("t5", "all"):
        preds = translate_t5(sources)
        results.append(evaluate_predictions(preds, references, "T5-Base"))

    # Summary table
    print("\n" + "="*65)
    print(f"{'Model':<22} {'BLEU':>8} {'chrF':>8} {'BERTScore':>10} {'CosSim':>8}")
    print("-"*65)
    for r in results:
        print(f"{r['model']:<22} {r['BLEU']:>8.4f} {r['chrF']:>8.2f} {r['BERTScore_F1']:>10.4f} {r['EmbeddingSim']:>8.4f}")
    print("="*65)

    # Best model
    best = max(results, key=lambda x: x["chrF"])
    print(f"\n🏆 Best model: {best['model']} (chrF: {best['chrF']})")


if __name__ == "__main__":
    main()
