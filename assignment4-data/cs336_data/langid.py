import fasttext
import os

_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        model_path = os.path.join(os.path.dirname(__file__), "assets", "lid.176.bin")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Language ID model not found at {model_path}")
        # Suppress fasttext warnings if possible, but load_model is main thing
        _MODEL = fasttext.load_model(model_path)
    return _MODEL


def identify_language(text: str) -> tuple[str, float]:
    model = _get_model()

    if not text or not text.strip():
        # Default/Fallback
        return "unknown", 0.0

    cleaned_text = text.replace("\n", " ")

    # helper for prediction
    # k=1 returns top 1
    labels, scores = model.predict(cleaned_text, k=1)

    if not labels:
        return "unknown", 0.0

    label = labels[0]
    score = scores[0]

    lang = label.replace("__label__", "")
    return lang, float(score)
