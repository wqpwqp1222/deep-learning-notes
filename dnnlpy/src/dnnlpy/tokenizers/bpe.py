import itertools as it
from collections import Counter
from typing import Self, override

from .base import Tokenizer

type Corpus = dict[tuple[str, ...], int]
type Symbols = tuple[str, ...]
type Pair = tuple[str, str]

__all__ = ['BPETokenizer']


def word2symbols(word: str) -> Symbols:
    """Represent a word as initial BPE symbols plus an end-of-word marker."""
    return tuple(word) + ('</w>',)


def get_pair_counts(corpus: Corpus) -> Counter[Pair]:
    """Count adjacent symbol pairs in a frequency-weighted corpus."""
    counts = Counter()
    for symbols, freq in corpus.items():
        for pair in it.pairwise(symbols):
            counts[pair] += freq
    return counts


def merge_pair(symbols: Symbols, pair: Pair) -> Symbols:
    """Merge every non-overlapping occurrence of pair in symbols."""
    merged = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
            merged.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            merged.append(symbols[i])
            i += 1
    return tuple(merged)


def merge_word(word: str, merges: list[Pair]) -> Symbols:
    """Apply learned BPE merges to a word."""
    symbols = word2symbols(word)
    for pair in merges:
        symbols = merge_pair(symbols, pair)
    return symbols


def train_bpe(
    corpus: Corpus,
    num_merges: int,
    min_frequency: int = 2,
) -> list[Pair]:
    """Learn BPE merge rules from a frequency-weighted symbolic corpus."""
    corpus = dict(corpus)
    merges = []

    for _ in range(num_merges):
        pair_counts = get_pair_counts(corpus)
        if not pair_counts:
            break

        best_pair, freq = pair_counts.most_common(1)[0]
        if freq < min_frequency:
            break

        merges.append(best_pair)

        new_corpus = Counter()
        for symbols, count in corpus.items():
            new_symbols = merge_pair(symbols, best_pair)
            new_corpus[new_symbols] = count
        corpus = new_corpus

    return merges


def build_bpe_vocab(
    alphabet: set[str],
    merges: list[Pair],
    special_tokens: list[str],
) -> dict[str, int]:
    """Build a BPE vocabulary from base symbols, merges, and special tokens."""
    tokens = set(alphabet)
    tokens.update(a + b for a, b in merges)

    vocab_tokens = special_tokens + sorted(tokens - set(special_tokens))
    return {token: i for i, token in enumerate(vocab_tokens)}


class BPETokenizer(Tokenizer):
    """Byte pair encoding tokenizer trained from whitespace-separated words."""

    def __init__(
        self,
        vocab: dict[str, int],
        merges: list[Pair],
        unk_token: str = '<unk>',
    ):
        """Create a BPE tokenizer from an existing vocabulary and merges.

        Args:
            vocab (dict[str, int]): Mapping from BPE tokens to IDs.
            merges (list[Pair]): Learned BPE merge rules.
            unk_token (str, default: '<unk>'): Token used for unknown BPE pieces.
        """
        self.merges = merges
        super().__init__(vocab, unk_token=unk_token)

    @override
    @classmethod
    def from_text(
        cls,
        text: str | list[str],
        vocab_size: int = 100,
        min_frequency: int = 2,
        unk_token: str = '<unk>',
    ) -> Self:
        """Train a BPE tokenizer from text.

        Args:
            text (str | list[str]): A single string or list of strings used as
                the training corpus.
            vocab_size (int, default: 100): Maximum vocabulary budget, including
                the unknown token.
            min_frequency (int, default: 2): Minimum pair frequency required to
                add a merge.
            unk_token (str, default: '<unk>'): Token used for unknown BPE pieces.
        """
        if isinstance(text, str):
            text = [text]

        word_freqs = Counter(word for line in text for word in line.split())
        corpus = {word2symbols(w): f for w, f in word_freqs.items()}

        # Vocab budget = unknown token + base alphabet + merged tokens.
        alphabet = {sym for symbols in corpus for sym in symbols}
        num_merges = max(0, vocab_size - len(alphabet) - 1)
        merges = train_bpe(corpus, num_merges, min_frequency)
        vocab = build_bpe_vocab(alphabet, merges, [unk_token])

        return cls(vocab, merges, unk_token)

    @override
    def encode(self, text: str) -> list[int]:
        """Encode text by applying learned BPE merges to each word.

        Args:
            text (str): Text to encode.
        """
        unk_id = self.vocab[self.unk_token]

        ids = []
        for word in text.split():
            for piece in merge_word(word, self.merges):
                ids.append(self.vocab.get(piece, unk_id))

        return ids

    @override
    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode BPE token IDs into text.

        Args:
            ids (list[int]): BPE token IDs to decode.
            skip_special_tokens (bool, default: True): Whether to omit special
                tokens from output.
        """
        if skip_special_tokens:
            special_tokens = set(self.special_tokens)
        else:
            special_tokens = set()

        tokens = []
        for i in ids:
            if self.id_to_token[int(i)] not in special_tokens:
                tokens.append(self.id_to_token[int(i)])

        return ''.join(tokens).replace('</w>', ' ').strip()
