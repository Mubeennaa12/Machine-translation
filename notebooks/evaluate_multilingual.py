"""
evaluation/evaluate_multilingual.py
─────────────────────────────────────────────────────────────────────────────
Evaluates NLLB-200 on English → 5 Indian languages using IndicMTEval.

Languages : Tamil · Hindi · Telugu · Kannada · Malayalam
Model      : facebook/nllb-200-distilled-600M
Metrics    : BLEU · chrF · BERTScore F1 · Cosine Similarity

Design rationale
────────────────
Primary evaluation (model selection, paper results) remains English → Tamil
because IndicMTEval has the richest human quality scores (MQM / DA) for Tamil.
This script adds NLLB capability demonstration across all five languages so
the final report can show multilingual breadth alongside the Tamil deep-dive.

Usage
─────
  # Evaluate all languages (≈200 samples each, default)
  python evaluate_multilingual.py

  # Single language fast check
  python evaluate_multilingual.py --lang Tamil --samples 50

  # Save results to JSON
  python evaluate_multilingual.py --output results/multilingual_eval.json
"""

import re
import json
import argparse
import numpy as np
import evaluate
import sacrebleu
from datasets import load_dataset
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
from bert_score import score as bert_score_fn
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Language → NLLB token + IndicMTEval filter string + BERTScore lang code
# ─────────────────────────────────────────────────────────────────────────────
LANG_CONFIG = {
    "Tamil":     {"token": "tam_Taml", "filter": "Tamil",     "bert_lang": "ta"},
    "Hindi":     {"token": "hin_Deva", "filter": "Hindi",     "bert_lang": "hi"},
    "Telugu":    {"token": "tel_Telu", "filter": "Telugu",    "bert_lang": "te"},
    "Kannada":   {"token": "kan_Knda", "filter": "Kannada",   "bert_lang": "kn"},
    "Malayalam": {"token": "mal_Mlym", "filter": "Malayalam", "bert_lang": "ml"},
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def load_lang_data(language: str, num_samples: int):
    """Load and filter IndicMTEval for a given language."""
    cfg = LANG_CONFIG[language]
    dataset = load_dataset("ai4bharat/IndicMTEval", split="test")
    subset = dataset.filter(lambda x: x["language"] == cfg["filter"])
    n = min(num_samples, len(subset))
    subset = subset.select(range(n))
    sources    = [preprocess(t) for t in subset["src"]]
    references = [preprocess(t) for t in subset["ref"]]
    print(f"  [{language}] {n} samples loaded.")
    return sources, references


# ─────────────────────────────────────────────────────────────────────────────
# Translation  (model + tokenizer passed in so they load only once)
# ─────────────────────────────────────────────────────────────────────────────
def translate_batch(sources, nllb_token, tokenizer, model, batch_size=8):
    predictions = []
    total = len(sources)
    for i in range(0, total, batch_size):
        batch = sources[i:i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(DEVICE)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(nllb_token),
                num_beams=4,
                max_length=256,
                early_stopping=True,
            )
        predictions.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
        done = min(i + batch_size, total)
        print(f"    Translated {done}/{total}", end="\r")
    print()
    return [preprocess(p) for p in predictions]


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(predictions, references, bert_lang: str):
    # Drop empty pairs
    pairs = [(p, r) for p, r in zip(predictions, references) if p.strip() and r.strip()]
    if not pairs:
        return {"bleu": 0, "chrf": 0, "bert_f1": 0, "cosine": 0, "n_valid": 0}

    preds, refs = zip(*pairs)
    preds, refs = list(preds), list(refs)

    # BLEU
    bleu_metric = evaluate.load("bleu")
    bleu_val = bleu_metric.compute(
        predictions=preds,
        references=[[r] for r in refs],
    )["bleu"]

    # chrF
    chrf_val = sacrebleu.corpus_chrf(preds, [refs]).score

    # BERTScore
    _, _, F1 = bert_score_fn(preds, refs, model_type="bert-base-multilingual-cased")
    bert_f1 = F1.mean().item()

    # Cosine similarity
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    ref_emb  = embed_model.encode(refs)
    pred_emb = embed_model.encode(preds)
    cos_sim  = cosine_similarity(ref_emb, pred_emb).diagonal().mean()

    return {
        "bleu":    round(bleu_val, 4),
        "chrf":    round(chrf_val, 2),
        "bert_f1": round(bert_f1, 4),
        "cosine":  round(float(cos_sim), 4),
        "n_valid": len(preds),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Multilingual MT Evaluation — NLLB-200")
    parser.add_argument("--lang",    default="all",
                        choices=list(LANG_CONFIG.keys()) + ["all"],
                        help="Language to evaluate (default: all)")
    parser.add_argument("--samples", type=int, default=200,
                        help="Max samples per language (default: 200)")
    parser.add_argument("--output",  default=None,
                        help="Optional JSON file to save results")
    args = parser.parse_args()

    languages = list(LANG_CONFIG.keys()) if args.lang == "all" else [args.lang]

    # Load model ONCE — reused across all languages
    print(f"\nLoading NLLB model on {DEVICE}...")
    tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    model     = AutoModelForSeq2SeqLM.from_pretrained(
        "facebook/nllb-200-distilled-600M"
    ).to(DEVICE)
    model.eval()
    print("Model ready ✓\n")

    all_results = {}

    for lang in languages:
        cfg = LANG_CONFIG[lang]
        print(f"{'─'*60}")
        print(f"  Language : {lang}  ({cfg['token']})")
        print(f"{'─'*60}")

        sources, references = load_lang_data(lang, args.samples)

        print(f"  Translating {len(sources)} sentences…")
        preds = translate_batch(sources, cfg["token"], tokenizer, model)

        print(f"  Computing metrics…")
        metrics = compute_metrics(preds, references, cfg["bert_lang"])
        metrics["language"] = lang
        metrics["nllb_token"] = cfg["token"]

        all_results[lang] = metrics
        print(f"  BLEU={metrics['bleu']:.4f}  chrF={metrics['chrf']:.2f}  "
              f"BERT={metrics['bert_f1']:.4f}  Cosine={metrics['cosine']:.4f}  "
              f"(n={metrics['n_valid']})\n")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"{'Language':<14} {'NLLB Token':<14} {'BLEU':>8} {'chrF':>8} {'BERTScore':>10} {'CosSim':>8} {'N':>6}")
    print("-" * 72)
    for lang, r in all_results.items():
        print(f"{lang:<14} {r['nllb_token']:<14} {r['bleu']:>8.4f} "
              f"{r['chrf']:>8.2f} {r['bert_f1']:>10.4f} {r['cosine']:>8.4f} {r['n_valid']:>6}")
    print("=" * 72)

    best_lang = max(all_results, key=lambda k: all_results[k]["chrf"])
    print(f"\n🏆 Best language result: {best_lang} (chrF: {all_results[best_lang]['chrf']})")
    print("   (Tamil is the primary evaluation language for model comparison.)\n")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    if args.output:
        import os
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"Results saved → {args.output}")


if __name__ == "__main__":
    main()
