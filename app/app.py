"""
English → Tamil Machine Translation App
Model: facebook/nllb-200-distilled-600M (Best Performing)
UI: Gradio with custom theme
"""

import gradio as gr
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
import torch
import re
import time

# ─────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────
MODEL_NAME = "facebook/nllb-200-distilled-600M"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model: {MODEL_NAME} on {DEVICE}...")
tokenizer = NllbTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()
print("Model loaded successfully!")


# ─────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ─────────────────────────────────────────
# Translation
# ─────────────────────────────────────────
def translate(text: str, num_beams: int = 4, max_length: int = 256) -> tuple[str, str]:
    if not text.strip():
        return "", "⚠️ Please enter some text to translate."

    start = time.time()
    clean = preprocess(text)

    inputs = tokenizer(
        clean,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(DEVICE)

    with torch.no_grad():
        translated = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("tam_Taml"),
            num_beams=num_beams,
            max_length=max_length,
            early_stopping=True,
        )

    result = tokenizer.decode(translated[0], skip_special_tokens=True)
    elapsed = time.time() - start

    info = f"✅ Translated {len(text.split())} words in {elapsed:.2f}s  |  Model: NLLB-200 (600M)  |  Device: {DEVICE.upper()}"
    return result, info


# ─────────────────────────────────────────
# Example Sentences
# ─────────────────────────────────────────
EXAMPLES = [
    ["The sun rises in the east and sets in the west."],
    ["Artificial intelligence is transforming the way we live and work."],
    ["She went to the market to buy fresh vegetables and fruits."],
    ["The children played happily in the park after school."],
    ["Please book a train ticket from Chennai to Coimbatore for tomorrow morning."],
    ["Climate change is one of the most pressing challenges of our time."],
]


# ─────────────────────────────────────────
# Custom CSS — Tamil script aesthetic, dark editorial theme
# ─────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --saffron:     #FF6B35;
    --deep-navy:   #0D1B2A;
    --slate:       #1C2E40;
    --card:        #162032;
    --border:      #263A50;
    --text:        #E8EDF2;
    --muted:       #7A94AA;
    --gold:        #E5A020;
    --teal:        #2EC4B6;
    --radius:      12px;
}

body, .gradio-container {
    background: var(--deep-navy) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text) !important;
}

/* ── Header ───────────────────────────── */
#header {
    text-align: center;
    padding: 48px 0 32px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 36px;
}
#header h1 {
    font-family: 'Playfair Display', serif !important;
    font-size: 2.6rem !important;
    font-weight: 700 !important;
    color: var(--text) !important;
    letter-spacing: -0.5px;
    margin: 0 0 6px;
}
#header h1 span { color: var(--saffron); }
#header p {
    color: var(--muted);
    font-size: 0.95rem;
    font-weight: 300;
    margin: 0;
}

/* ── Language badge ───────────────────── */
#lang-bar {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 14px;
    margin-bottom: 28px;
}
.lang-chip {
    background: var(--slate);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 6px 18px;
    font-size: 0.85rem;
    font-weight: 500;
    color: var(--text);
}
.arrow-chip {
    color: var(--saffron);
    font-size: 1.2rem;
}

/* ── Textboxes ────────────────────────── */
.input-box textarea, .output-box textarea {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    line-height: 1.65 !important;
    padding: 16px !important;
    resize: vertical !important;
    transition: border-color 0.2s;
}
.input-box textarea:focus, .output-box textarea:focus {
    border-color: var(--saffron) !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(255,107,53,0.12) !important;
}
.output-box textarea {
    border-color: var(--teal) !important;
    background: rgba(46,196,182,0.04) !important;
}
label span {
    color: var(--muted) !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

/* ── Translate button ─────────────────── */
#translate-btn {
    background: linear-gradient(135deg, var(--saffron), #E84A1A) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    padding: 14px 0 !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: opacity 0.2s, transform 0.1s !important;
    letter-spacing: 0.02em;
}
#translate-btn:hover  { opacity: 0.88 !important; transform: translateY(-1px) !important; }
#translate-btn:active { transform: translateY(0) !important; }

/* ── Status bar ───────────────────────── */
#status-box textarea {
    background: var(--slate) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--teal) !important;
    font-size: 0.8rem !important;
    font-family: 'DM Mono', monospace !important;
    padding: 10px 14px !important;
    min-height: unset !important;
    resize: none !important;
}

/* ── Advanced accordion ───────────────── */
.gr-accordion {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

/* ── Sliders ──────────────────────────── */
input[type=range] { accent-color: var(--saffron) !important; }

/* ── Examples ─────────────────────────── */
.gr-examples table {
    background: var(--card) !important;
    border-radius: var(--radius) !important;
}
.gr-examples td {
    color: var(--muted) !important;
    border-color: var(--border) !important;
    font-size: 0.88rem !important;
}
.gr-examples tr:hover td { color: var(--text) !important; }

/* ── Metric cards ─────────────────────── */
#metrics {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin: 28px 0;
}
.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px;
    text-align: center;
}
.metric-card .val {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    color: var(--gold);
    display: block;
}
.metric-card .lbl {
    font-size: 0.72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 4px;
    display: block;
}

/* ── Footer ───────────────────────────── */
#footer {
    text-align: center;
    padding: 28px 0 12px;
    border-top: 1px solid var(--border);
    margin-top: 40px;
    color: var(--muted);
    font-size: 0.8rem;
}
"""

# ─────────────────────────────────────────
# Build Gradio UI
# ─────────────────────────────────────────
with gr.Blocks(css=CSS, title="English → Tamil Translator") as demo:

    # Header
    gr.HTML("""
    <div id="header">
        <h1>English → <span>Tamil</span> Translator</h1>
        <p>Powered by NLLB-200 · Best-in-class multilingual neural machine translation</p>
    </div>
    <div id="lang-bar">
        <span class="lang-chip">🇬🇧 English</span>
        <span class="arrow-chip">→</span>
        <span class="lang-chip">🇮🇳 தமிழ் (Tamil)</span>
    </div>
    """)

    # Model performance metrics
    gr.HTML("""
    <div id="metrics">
        <div class="metric-card"><span class="val">0.142</span><span class="lbl">BLEU Score</span></div>
        <div class="metric-card"><span class="val">41.3</span><span class="lbl">chrF Score</span></div>
        <div class="metric-card"><span class="val">0.618</span><span class="lbl">BERTScore F1</span></div>
        <div class="metric-card"><span class="val">0.731</span><span class="lbl">Cosine Sim</span></div>
    </div>
    """)

    # Main translation panel
    with gr.Row():
        with gr.Column(scale=1):
            src_text = gr.Textbox(
                label="English Source",
                placeholder="Type or paste English text here...",
                lines=8,
                elem_classes=["input-box"],
            )
        with gr.Column(scale=1):
            tgt_text = gr.Textbox(
                label="Tamil Translation  ·  தமிழ் மொழிபெயர்ப்பு",
                lines=8,
                interactive=False,
                elem_classes=["output-box"],
            )

    translate_btn = gr.Button("⟶  Translate", elem_id="translate-btn")
    status_box = gr.Textbox(
        label="",
        interactive=False,
        lines=1,
        elem_id="status-box",
        elem_classes=["status-box"],
    )

    # Advanced settings
    with gr.Accordion("⚙️  Advanced Settings", open=False):
        with gr.Row():
            num_beams = gr.Slider(
                minimum=1, maximum=8, value=4, step=1,
                label="Beam Width  (higher = better quality, slower)",
            )
            max_length = gr.Slider(
                minimum=64, maximum=512, value=256, step=32,
                label="Max Output Tokens",
            )

    # Examples
    gr.Examples(
        examples=EXAMPLES,
        inputs=src_text,
        label="📌 Try an Example",
    )

    # Footer
    gr.HTML("""
    <div id="footer">
        Model: facebook/nllb-200-distilled-600M &nbsp;·&nbsp;
        Dataset: ai4bharat/IndicMTEval &nbsp;·&nbsp;
        Built with Gradio &nbsp;·&nbsp;
        English → Tamil (tam_Taml)
    </div>
    """)

    # Wiring
    translate_btn.click(
        fn=translate,
        inputs=[src_text, num_beams, max_length],
        outputs=[tgt_text, status_box],
    )
    src_text.submit(
        fn=translate,
        inputs=[src_text, num_beams, max_length],
        outputs=[tgt_text, status_box],
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
