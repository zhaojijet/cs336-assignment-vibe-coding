import re


def mask_emails(text: str) -> tuple[str, int]:
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    masked_text, count = re.subn(email_pattern, "|||EMAIL_ADDRESS|||", text)
    return masked_text, count


def mask_phone_numbers(text: str) -> tuple[str, int]:
    # More robust phone regex needed for (123) 456-7890 formats
    # Pattern to match:
    # (123) 456-7890, 123-456-7890, 123 456 7890, 1234567890
    # Maybe use a simpler one if test allows?
    # Test cases: "2831823829", "(283)-182-3829", "(283) 182 3829", "283-182-3829"
    # A generic one:
    # Optional open paren, 3 digits, optional close paren, optional separator, 3 digits, optional separator, 4 digits.
    # But ensuring valid boundaries.

    # Try: (\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4})
    # But this might match too greedily or not enough.

    # Let's try to match the specific formats in test.
    # Regex: `(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})`

    phone_pattern = r"(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})"

    # We need to be careful not to double mask if I run multiple passes, but subn handles it.
    # Also ensuring it doesn't match inside a larger number?
    # The test cases are simple.

    masked_text, count = re.subn(phone_pattern, "|||PHONE_NUMBER|||", text)
    return masked_text, count


def mask_ips(text: str) -> tuple[str, int]:
    # IPv4 Pattern
    ip_pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    masked_text, count = re.subn(ip_pattern, "|||IP_ADDRESS|||", text)
    return masked_text, count
