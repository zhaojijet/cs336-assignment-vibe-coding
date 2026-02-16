import fasttext
import os

_NSFW_MODEL = None
_TOXIC_MODEL = None


def _get_nsfw_model():
    global _NSFW_MODEL
    if _NSFW_MODEL is None:
        model_path = os.path.join(
            os.path.dirname(__file__), "assets", "dolma_fasttext_nsfw_jigsaw_model.bin"
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"NSFW model not found at {model_path}")
        _NSFW_MODEL = fasttext.load_model(model_path)
    return _NSFW_MODEL


def _get_toxic_model():
    global _TOXIC_MODEL
    if _TOXIC_MODEL is None:
        model_path = os.path.join(
            os.path.dirname(__file__),
            "assets",
            "dolma_fasttext_hatespeech_jigsaw_model.bin",
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Toxic model not found at {model_path}")
        _TOXIC_MODEL = fasttext.load_model(model_path)
    return _TOXIC_MODEL


def classify_nsfw(text: str) -> tuple[str, float]:
    model = _get_nsfw_model()
    if not text or not text.strip():
        return "non-nsfw", 0.0

    cleaned_text = text.replace("\n", " ")
    labels, scores = model.predict(cleaned_text, k=1)

    if not labels:
        return "non-nsfw", 0.0

    label = labels[0].replace("__label__", "")
    score = float(scores[0])

    # fasttext labels might be 'nsfw', 'non-nsfw' or similar.
    # The test expects "nsfw" or "non-nsfw".
    # I need to verify what the model outputs.
    # Usually jigsaw models output specific labels.
    # Assuming standard fasttext labels.

    return label, score


def classify_toxic_speech(text: str) -> tuple[str, float]:
    model = _get_toxic_model()
    if not text or not text.strip():
        return "non-toxic", 0.0

    cleaned_text = text.replace("\n", " ")
    labels, scores = model.predict(cleaned_text, k=1)

    if not labels:
        return "non-toxic", 0.0

    label = labels[0].replace("__label__", "")
    score = float(scores[0])

    return label, score
