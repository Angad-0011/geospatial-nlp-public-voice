import types

import pytest

import preprocess

from utils_io import load_df, save_df



class DummyToken:
    def __init__(self, text, lemma):
        self.text = text
        self.lemma_ = lemma


class DummyDoc:
    def __init__(self, text):
        self.text = text
        self.ents = []
        words = text.split()
        self._tokens = [DummyToken(word, word.rstrip('s')) for word in words]

    def __iter__(self):
        return iter(self._tokens)


def dummy_nlp(text):
    return DummyDoc(text)


def test_clean_text_removes_urls():
    text = "Check https://example.com for details!"
    cleaned = preprocess.clean_text(text)
    assert "http" not in cleaned
    assert cleaned.startswith("check")


def test_lemmatize_uses_stub(monkeypatch):
    preprocess.STOPWORDS = {"and"}
    monkeypatch.setattr(preprocess, "get_nlp", lambda: dummy_nlp)
    lemma = preprocess.lemmatize("roads and services")
    assert lemma == "road service"
