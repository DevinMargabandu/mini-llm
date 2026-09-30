from __future__ import annotations

from minillm.tokenizer.bpe import BPETokenizer

CORPUS = (
    "the quick brown fox jumps over the lazy dog. "
    "the dog barks. the fox runs away quickly! "
) * 20


def make_trained_tokenizer(vocab_size: int = 300) -> BPETokenizer:
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=vocab_size)
    return tok


def test_round_trip_on_training_text():
    tok = make_trained_tokenizer()
    ids = tok.encode(CORPUS)
    assert tok.decode(ids) == CORPUS


def test_round_trip_on_unseen_text():
    tok = make_trained_tokenizer()
    text = "the zebra jumps over a fence, does it not?"
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_round_trip_on_unicode_and_punctuation():
    tok = make_trained_tokenizer()
    text = "Héllo, wörld! 🚀 costs $3.50 — ok?"
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_empty_string():
    tok = make_trained_tokenizer()
    assert tok.encode("") == []
    assert tok.decode([]) == ""


def test_vocab_size_matches_requested_upper_bound():
    tok = make_trained_tokenizer(vocab_size=300)
    assert tok.vocab_size <= 300
    assert tok.vocab_size > 256  # some merges should have been learned


def test_compression_beats_raw_bytes():
    tok = make_trained_tokenizer(vocab_size=300)
    text = "the quick brown fox jumps over the lazy dog."
    ids = tok.encode(text)
    assert len(ids) < len(text.encode("utf-8"))


def test_special_tokens_round_trip():
    tok = make_trained_tokenizer()
    tok.add_special_tokens(["<|endoftext|>"])
    eot = tok.special_tokens["<|endoftext|>"]
    ids = tok.encode("hello") + [eot] + tok.encode("world")
    assert tok.decode(ids) == "hello<|endoftext|>world"


def test_save_and_load_round_trip(tmp_path):
    tok = make_trained_tokenizer()
    tok.add_special_tokens(["<|endoftext|>"])
    path = tmp_path / "tok.json"
    tok.save(path)

    loaded = BPETokenizer.load(path)
    text = "the fox and the dog are friends."
    assert loaded.encode(text) == tok.encode(text)
    assert loaded.decode(tok.encode(text)) == text
    assert loaded.special_tokens == tok.special_tokens
