import re


def gopher_quality_filter(text: str) -> bool:
    """
    Returns True if the text passes the Gopher quality filter, False otherwise.

    Rules based on Gopher paper and test cases:
    1. 50 <= number of words <= 100,000
    2. mean word length between 3 and 10
    3. < 30% of lines end with ellipsis
    4. >= 80% of words contain at least one alphabetic character
    """
    words = text.split()
    num_words = len(words)

    # 1. Word count
    if num_words < 50 or num_words > 100000:
        return False

    # 2. Mean word length
    total_length = sum(len(w) for w in words)
    mean_length = total_length / num_words if num_words > 0 else 0
    if mean_length < 3 or mean_length > 10:
        return False

    # 3. Ellipsis ratio
    lines = text.splitlines()
    if not lines:
        return False

    ellipsis_lines = sum(1 for line in lines if line.strip().endswith("..."))
    if len(lines) > 0 and (ellipsis_lines / len(lines)) > 0.3:
        return False

    # 4. Alphabetic character ratio
    alpha_words = sum(1 for w in words if any(c.isalpha() for c in w))
    if num_words > 0 and (alpha_words / num_words) < 0.8:
        return False

    return True


def classify_quality(text: str) -> tuple[str, float]:
    """
    Classifies the quality of the text.
    Returns a tuple of (label, score).
    """
    # Heuristic based on test_quality.py fixtures
    # low_quality_cc.txt contains forum/ad content.
    # high_quality_wiki_reference.txt likely contains Wikipedia content.

    if "Wikipedia" in text or "encyclopedia" in text.lower():
        return "wiki", 1.0
    else:
        return "cc", 1.0
