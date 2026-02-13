import regex as re
from typing import BinaryIO
import functools

# Regular expression used for GPT-2 pre-tokenization
GPT2_SPLIT_PATTERN = (
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)


class BPE:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.special_token_ids = {
            token: idx
            for idx, token_bytes in vocab.items()
            if (token := token_bytes.decode("utf-8", errors="ignore"))
            in self.special_tokens
        }

        # Build the inverse vocab for encoding
        self.encoder = {v: k for k, v in vocab.items()}
        # Build the byte encoder for initial byte-to-unicode mapping
        self.byte_encoder = self.bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        # Build the merges dictionary
        # merges is list[tuple[bytes, bytes]]. bpe uses strings (mapped chars).
        # We must convert the byte merges to string merges using the byte_encoder.
        self.bpe_ranks = {}
        for i, (b1, b2) in enumerate(merges):
            s1 = "".join(self.byte_encoder[b] for b in b1)
            s2 = "".join(self.byte_encoder[b] for b in b2)
            self.bpe_ranks[(s1, s2)] = i
        self.cache = {}

    @functools.lru_cache()
    def bytes_to_unicode(self):
        """
        Returns a mapping between every possible byte (an integer from 0 to 255) to a
        printable unicode string character representation.
        """
        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        return {b: chr(c) for b, c in zip(bs, cs)}

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]

        word = tuple(token)
        pairs = self.get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            else:
                pairs = self.get_pairs(word)
        self.cache[token] = word
        return word

    def get_pairs(self, word):
        """Return set of symbol pairs in a word.
        Word is represented as tuple of symbols (symbols being variable-length strings).
        """
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def encode(self, text: str) -> list[int]:
        bpe_tokens = []

        # Ensure special tokens are sorted by length descending to match longest first
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)

        if sorted_special_tokens:
            pattern = "(" + "|".join(re.escape(k) for k in sorted_special_tokens) + ")"
            parts = re.split(pattern, text)
        else:
            parts = [text]

        for part in parts:
            if part in self.special_tokens:
                encoded_part = part.encode("utf-8")
                if encoded_part in self.encoder:
                    bpe_tokens.append(self.encoder[encoded_part])
                else:
                    pass
            else:
                if not part:
                    continue
                # Split by GPT2 pattern
                for token in re.findall(GPT2_SPLIT_PATTERN, part):
                    token_bytes = token.encode("utf-8")
                    token_translated = "".join(
                        self.byte_encoder[b] for b in token_bytes
                    )

                    # Run BPE
                    word_bpe_tokens = self.bpe(token_translated)

                    # Map back to IDs
                    for bpe_token in word_bpe_tokens:
                        original_bytes = bytes(self.byte_decoder[c] for c in bpe_token)
                        if original_bytes in self.encoder:
                            bpe_tokens.append(self.encoder[original_bytes])
                        else:
                            for b in original_bytes:
                                bpe_tokens.append(self.encoder[bytes([b])])
        return bpe_tokens

    def decode(self, ids: list[int]) -> str:
        text_bytes = b"".join([self.vocab[idx] for idx in ids])
        return text_bytes.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: functools.partial) -> list[int]:
        for chunk in iterable:
            yield from self.encode(chunk)


def train_bpe(
    input_path: str | str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Pre-tokenize
    if special_tokens:
        # Use a non-capturing group for the split pattern to avoid empty strings if we don't want them,
        # but re.split with capturing group (parentheses) returns the delimiters.
        # We WANT the delimiters (special tokens) to be in the list.
        # Check if special_tokens need escaping.
        pattern = "(" + "|".join(re.escape(k) for k in special_tokens) + ")"
        parts = re.split(pattern, text)
    else:
        parts = [text]

    words = []
    for part in parts:
        if part in special_tokens:
            continue
        # Check for empty or whitespace-only parts?
        # No, whitespace is part of BPE.
        if not part:
            continue
        words.extend(re.findall(GPT2_SPLIT_PATTERN, part))

    # Init stats
    # Convert each word to a tuple of bytes
    # But wait, GPT-2 BPE operates on *bytes* mapped to unicode chars for reversibility.
    # We should follow that to match GPT-2 perfectly.

    # 1. Byte encoder
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    byte_encoder = {b: chr(c) for b, c in zip(bs, cs)}
    byte_decoder = {v: k for k, v in byte_encoder.items()}

    # Count words
    word_counts = {}
    for word in words:
        word_bytes = word.encode("utf-8")
        word_chars = tuple(byte_encoder[b] for b in word_bytes)
        word_counts[word_chars] = word_counts.get(word_chars, 0) + 1

    # Init vocab
    # Base vocab is all the characters present in the data?
    # Usually we initialize with all 256 bytes to ensure full coverage.
    vocab = {i: bytes([i]) for i in range(256)}
    # Add special tokens
    for i, st in enumerate(special_tokens):
        vocab[len(vocab)] = st.encode("utf-8")

    merges = []

    # We need to map current vocab to bytes for the return value
    # But the active vocab during training is counting pairs.

    # Current vocab loop
    while len(vocab) < vocab_size:
        pairs = {}
        for word, count in word_counts.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pairs[pair] = pairs.get(pair, 0) + count

        if not pairs:
            break

        # Deterministic tie breaking: High count, then lexicographically largest byte pair
        # This matches observed behavior in reference (e.g. 'l' > 'c', 'm' > 'd')
        def get_stat_key(p):
            b1 = bytes(byte_decoder[c] for c in p[0])
            b2 = bytes(byte_decoder[c] for c in p[1])
            return (pairs[p], b1, b2)

        best = max(pairs, key=get_stat_key)
        merges.append(best)

        # best is (char1, char2) where chars are from byte_encoder
        # We need to construct the new byte sequence
        # But wait, the return type for merges is tuple[bytes, bytes]
        # And the vocab is dict[int, bytes]

        # We need to maintain a mapping from 'char' (unicode) to 'bytes' (original)
        # Actually we can just reconstruct it at the end or track it.

        # Let's update the word counts
        new_word_counts = {}
        bigram = best
        out_token = bigram[0] + bigram[1]

        # Update vocab
        # We need to convert the unicode chars back to bytes to store in vocab
        # This checks if the new token is in special tokens? No.

        # We need a way to convert the unicode string `out_token` back to bytes
        byte_decoder = {v: k for k, v in byte_encoder.items()}
        new_token_bytes = bytes(byte_decoder[c] for c in out_token)
        vocab[len(vocab)] = new_token_bytes

        # Update counts
        for word, count in word_counts.items():
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(bigram[0], i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if (
                    i < len(word) - 1
                    and word[i] == bigram[0]
                    and word[i + 1] == bigram[1]
                ):
                    new_word.append(out_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_counts[tuple(new_word)] = (
                new_word_counts.get(tuple(new_word), 0) + count
            )
        word_counts = new_word_counts

    # Convert merges to bytes
    byte_decoder = {v: k for k, v in byte_encoder.items()}
    final_merges = []
    for m1, m2 in merges:
        b1 = bytes(byte_decoder[c] for c in m1)
        b2 = bytes(byte_decoder[c] for c in m2)
        final_merges.append((b1, b2))

    return vocab, final_merges
