"""
English → Indian Languages Machine Translation App
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
# Supported Languages
# ─────────────────────────────────────────
LANGUAGES = {
    "Tamil": "tam_Taml",
    "Hindi": "hin_Deva",
    "Telugu": "tel_Telu",
    "Kannada": "kan_Knda",
    "Malayalam": "mal_Mlym"
}


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
def translate(text: str, target_lang: str, num_beams: int = 4, max_length: int = 256):

    if not text.strip():
        return "", "⚠️ Please enter some text to translate."

    start = time.time()
    clean = preprocess(text)

    lang_code = LANGUAGES[target_lang]

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
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(lang_code),
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
# Custom CSS (unchanged)
# ─────────────────────────────────────────
CSS = """
body, .gradio-container {
    background: #0D1B2A !important;
    font-family: 'DM Sans', sans-serif !important;
    color: white !important;
}
"""


# ─────────────────────────────────────────
# Build Gradio UI
# ─────────────────────────────────────────
with gr.Blocks(css=CSS, title="English → Indian Languages Translator") as demo:

    gr.Markdown(
        """
# 🌏 English → Indian Languages Translator
Powered by **NLLB-200 multilingual translation model**
"""
    )

    with gr.Row():

        src_text = gr.Textbox(
            label="English Input",
            placeholder="Enter English sentence...",
            lines=6,
        )

        tgt_text = gr.Textbox(
            label="Translation",
            lines=6,
            interactive=False,
        )

    target_lang = gr.Dropdown(
        choices=list(LANGUAGES.keys()),
        value="Tamil",
        label="Target Language"
    )

    translate_btn = gr.Button("Translate")

    status_box = gr.Textbox(
        label="Status",
        interactive=False,
        lines=1,
    )

    with gr.Accordion("Advanced Settings", open=False):

        num_beams = gr.Slider(
            minimum=1,
            maximum=8,
            value=4,
            step=1,
            label="Beam Width",
        )

        max_length = gr.Slider(
            minimum=64,
            maximum=512,
            value=256,
            step=32,
            label="Max Output Tokens",
        )

    gr.Examples(
        examples=EXAMPLES,
        inputs=src_text,
        label="Example Sentences",
    )

    translate_btn.click(
        fn=translate,
        inputs=[src_text, target_lang, num_beams, max_length],
        outputs=[tgt_text, status_box],
    )

    src_text.submit(
        fn=translate,
        inputs=[src_text, target_lang, num_beams, max_length],
        outputs=[tgt_text, status_box],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)