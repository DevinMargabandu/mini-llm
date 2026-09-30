"""A from-scratch byte-level BPE tokenizer, in the spirit of GPT-2's tokenizer.

Text is first encoded as raw UTF-8 bytes (so any input is representable with
no "unknown token" cases), then adjacent byte/token pairs are iteratively
merged based on frequency statistics collected from a training corpus.
"""
from __future__ import annotations

import json
import regex as _regex_unused  # noqa: F401  (kept optional; see _SPLIT_PATTERN note)
from collections import Counter
from pathlib import Path
from typing import Iterable

# GPT-2 style pre-tokenization regex: split text into chunks (roughly words,
# punctuation runs, and whitespace) *before* BPE-merging within each chunk.
# This keeps merges from crossing word boundaries.
_SPLIT_PATTERN = _regex_unused.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

Pair = tuple[int, int]


def _get_pair_counts(token_seqs: list[list[int]]) -> Counter[Pair]:
    counts: Counter[Pair] = Counter()
    for seq in token_seqs:
        for a, b in zip(seq, seq[1:]):
            counts[(a, b)] += 1
    return counts


def _merge(seq: list[int], pair: Pair, new_id: int) -> list[int]:
    out: list[int] = []
    i = 0
    n = len(seq)
    while i < n:
        if i < n - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


class _IncrementalPairIndex:
    """Tracks pair frequencies across many (weighted) chunks, and which
    chunks contain each pair, so a merge only needs to revisit the handful
    of chunks it actually touches instead of rescanning the whole corpus.

    This turns naive O(num_merges * corpus_size) BPE training into
    something close to O(corpus_size + num_merges * affected_chunk_size),
    which is what makes training on a real corpus (millions of characters)
    tractable in pure Python.
    """

    def __init__(self, chunks: list[list[int]], weights: list[int]) -> None:
        self.chunks = chunks
        self.weights = weights
        self.pair_counts: Counter[Pair] = Counter()
        self.pair_to_chunks: dict[Pair, set[int]] = {}
        for idx, chunk in enumerate(chunks):
            self._index_chunk(idx)

    def _chunk_pairs(self, idx: int) -> Iterable[Pair]:
        chunk = self.chunks[idx]
        return zip(chunk, chunk[1:])

    def _index_chunk(self, idx: int) -> None:
        w = self.weights[idx]
        for pair in self._chunk_pairs(idx):
            self.pair_counts[pair] += w
            self.pair_to_chunks.setdefault(pair, set()).add(idx)

    def _deindex_chunk(self, idx: int) -> None:
        w = self.weights[idx]
        for pair in self._chunk_pairs(idx):
            self.pair_counts[pair] -= w
            if self.pair_counts[pair] <= 0:
                del self.pair_counts[pair]

    def most_common_pair(self) -> tuple[Pair, int] | None:
        if not self.pair_counts:
            return None
        return max(self.pair_counts.items(), key=lambda kv: kv[1])

    def apply_merge(self, pair: Pair, new_id: int) -> None:
        affected = self.pair_to_chunks.pop(pair, set())
        for idx in affected:
            self._deindex_chunk(idx)
            self.chunks[idx] = _merge(self.chunks[idx], pair, new_id)
            self._index_chunk(idx)


class BPETokenizer:
    """Byte-level Byte-Pair Encoding tokenizer.

    Vocabulary layout:
      - ids [0, 256)      : the 256 raw byte values
      - ids [256, 256+k)  : learned merges, in the order they were learned
      - trailing ids       : special tokens (e.g. <|endoftext|>)
    """

    def __init__(self) -> None:
        self.merges: dict[Pair, int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special_tokens: dict[str, int] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.special_tokens)

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        if vocab_size < 256:
            raise ValueError("vocab_size must be >= 256 (one id per raw byte)")
        num_merges = vocab_size - 256

        # Identical chunks (e.g. the word "the") recur constantly in real
        # text. Dedupe them and carry a frequency weight instead of running
        # the merge loop over every occurrence separately.
        chunk_counts = Counter(_SPLIT_PATTERN.findall(text))
        unique_chunks = list(chunk_counts.keys())
        weights = list(chunk_counts.values())
        token_seqs: list[list[int]] = [list(chunk.encode("utf-8")) for chunk in unique_chunks]

        index = _IncrementalPairIndex(token_seqs, weights)

        next_id = 256
        for i in range(num_merges):
            best = index.most_common_pair()
            if best is None:
                break
            best_pair, best_count = best
            if best_count < 2:
                break  # no more pairs worth merging

            index.apply_merge(best_pair, next_id)
            self.merges[best_pair] = next_id
            self.vocab[next_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            if verbose and (i % 200 == 0 or i == num_merges - 1):
                print(
                    f"merge {i + 1}/{num_merges}: {best_pair} -> {next_id} "
                    f"({self.vocab[next_id]!r}) had {best_count} occurrences"
                )
            next_id += 1

    def add_special_tokens(self, tokens: Iterable[str]) -> None:
        next_id = len(self.vocab) + len(self.special_tokens)
        for tok in tokens:
            if tok not in self.special_tokens:
                self.special_tokens[tok] = next_id
                next_id += 1

    # ------------------------------------------------------------------ #
    # Encode / decode
    # ------------------------------------------------------------------ #
    def _encode_chunk(self, chunk_bytes: bytes) -> list[int]:
        ids = list(chunk_bytes)
        while len(ids) >= 2:
            pair_counts = _get_pair_counts([ids])
            # Pick the eligible pair that was learned earliest during training
            # (lowest merge id), matching the order merges were created in.
            candidate = min(
                pair_counts.keys(),
                key=lambda p: self.merges.get(p, float("inf")),
                default=None,
            )
            if candidate is None or candidate not in self.merges:
                break
            ids = _merge(ids, candidate, self.merges[candidate])
        return ids

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        chunks = _SPLIT_PATTERN.findall(text)
        ids: list[int] = []
        for chunk in chunks:
            ids.extend(self._encode_chunk(chunk.encode("utf-8")))
        return ids

    def decode(self, ids: list[int]) -> str:
        inv_special = {v: k for k, v in self.special_tokens.items()}
        parts: list[bytes] = []
        text_parts: list[str] = []
        for i in ids:
            if i in inv_special:
                if parts:
                    text_parts.append(b"".join(parts).decode("utf-8", errors="replace"))
                    parts = []
                text_parts.append(inv_special[i])
            else:
                parts.append(self.vocab[i])
        if parts:
            text_parts.append(b"".join(parts).decode("utf-8", errors="replace"))
        return "".join(text_parts)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        path = Path(path)
        data = {
            "merges": [[list(pair), new_id] for pair, new_id in self.merges.items()],
            "special_tokens": self.special_tokens,
        }
        path.write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        data = json.loads(Path(path).read_text())
        tok = cls()
        for (a, b), new_id in data["merges"]:
            tok.merges[(a, b)] = new_id
            tok.vocab[new_id] = tok.vocab[a] + tok.vocab[b]
        tok.special_tokens = {k: int(v) for k, v in data["special_tokens"].items()}
        return tok
