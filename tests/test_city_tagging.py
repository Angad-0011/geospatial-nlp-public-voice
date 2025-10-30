import preprocess
from utils_io import load_df, save_df


class Entity:
    def __init__(self, text, label_):
        self.text = text
        self.label_ = label_


class Doc:
    def __init__(self, ents):
        self.ents = ents


def test_guess_city_regex(monkeypatch):
    monkeypatch.setattr(preprocess, "get_nlp", lambda: lambda text: Doc([]))
    city = preprocess.guess_city("traffic issues rising in Bengaluru")
    assert city.lower() == "bengaluru"


def test_guess_city_ner(monkeypatch):
    monkeypatch.setattr(preprocess, "get_nlp", lambda: lambda text: Doc([Entity("Hyderabad", "GPE")]))
    city = preprocess.guess_city("Metro expansion announced", existing=None)
    assert city == "Hyderabad"
