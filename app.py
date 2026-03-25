import gradio as gr
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # Prevent freezes in headless container

import string
from lime.lime_text import LimeTextExplainer
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# -------------------------------
# Model Load
# -------------------------------
MODEL_PATH = "./distilbert_model"

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

class_names = ["Human", "AI"]

# -------------------------------
# Predictor
# -------------------------------
def predictor(texts):
    inputs = tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
    return probs

# -------------------------------
# LIME
# -------------------------------
explainer = LimeTextExplainer(
    class_names=class_names,
    split_expression=lambda text: text.split(" ")
)

def generate_lime(user_text):
    if not user_text:
        return "Enter text"

    exp = explainer.explain_instance(user_text, predictor, num_features=10, num_samples=25)
    weights = {w.lower().strip(string.punctuation): s for w, s in exp.as_list()}

    probs = predictor([user_text])[0]
    pred = class_names[np.argmax(probs)]

    html = []
    for w in user_text.split():
        score = weights.get(w.lower().strip(string.punctuation), 0)
        color = "rgba(0,128,0,0.3)" if score > 0 else "rgba(255,0,0,0.3)"
        html.append(f'<span style="background:{color};padding:2px">{w}</span>')

    return f"<b>{pred}</b><br>" + " ".join(html)

# -------------------------------
# SHAP (LAZY LOAD)
# -------------------------------
shap_explainer = None
shap = None

def get_shap_explainer():
    global shap_explainer, shap
    if shap is None:
        import shap  # Lazy import prevents startup freeze
    if shap_explainer is None:
        masker = shap.maskers.Text(tokenizer, mask_token="<unk>")
        shap_explainer = shap.Explainer(predictor, masker)
    return shap_explainer

def generate_shap(user_text):
    if not user_text:
        return "Enter text"
    try:
        expl = get_shap_explainer()
        # limit input to prevent freeze
        shap_values = expl([user_text[:150]])[0]

        words = shap_values.data
        values = shap_values.values
        if len(values.shape) > 1:
            values = values[:, 1]

        html = []
        max_val = max(abs(values).max(), 1)
        for w, v in zip(words, values):
            alpha = min(0.8, abs(v)/max_val*0.8 + 0.2)
            color = f"rgba(255,0,0,{alpha})" if v > 0 else f"rgba(0,128,0,{alpha})"
            html.append(f'<span style="background:{color};padding:2px">{w}</span>')
        return " ".join(html)
    except Exception:
        return "SHAP is loading... try again"

# -------------------------------
# Combined
# -------------------------------
def run_all(text):
    lime_result = generate_lime(text)
    shap_result = generate_shap(text)
    return lime_result, shap_result

# -------------------------------
# UI
# -------------------------------
with gr.Blocks() as demo:
    gr.Markdown("# DistilBERT AI vs Human Detector")
    text = gr.Textbox(lines=5, label="Enter text")

    with gr.Row():
        lime_out = gr.HTML(label="LIME")
        shap_out = gr.HTML(label="SHAP")

    text.change(run_all, inputs=text, outputs=[lime_out, shap_out])

# -------------------------------
# Launch
# -------------------------------
demo.launch(ssr_mode=False)
