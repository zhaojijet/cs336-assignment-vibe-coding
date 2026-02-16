from resiliparse.extract.html2text import extract_plain_text


def extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    if not html_bytes:
        return ""
    try:
        # resiliparse in this environment doesn't support link_texts argument?
        # And it might work better on decoded string?
        # Let's try to decode if possible, or pass bytes if that was the issue.
        # But debug script showed decoding works.
        try:
            text = html_bytes.decode("utf-8")
        except UnicodeDecodeError:
            # Fallback to loose decoding
            text = html_bytes.decode("utf-8", errors="replace")

        # extract_plain_text supports string/bytes.
        # Removing link_texts argument which caused error.
        return extract_plain_text(text, main_content=False, alt_texts=True)
    except Exception:
        return ""
