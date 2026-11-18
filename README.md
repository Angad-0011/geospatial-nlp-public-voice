# Geospatial NLP Public Voice

A Python prototype for exploring how local issues appear in Reddit discussions and news coverage. It tags documents by city, builds a searchable corpus, and presents topic and sentiment signals alongside semantic search. Results describe the corpus, not public opinion as a whole.

## Data sources

- **Sample mode:** reads the two checked-in Parquet fixtures, `data/sample/sample_reddit.parquet` and `data/sample/sample_news.parquet`. Their record-level provenance and redistribution rights are not documented here; don't assume these files are synthetic or licensed for reuse.
- **Live mode:** runs a configured Reddit search and fetches news from configured sources, which default to The Hindu, The Indian Express, and Deccan Herald. Use only sources you're authorized to collect and reuse. The Reddit collector relies on the legacy `snscrape` integration; live access is not verified.

## Quickstart

Start with Python 3.11, the version the repository's Dockerfile targets. The first run may download NLP models and NLTK resources even when live scraping is off.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Sample pipeline and tests

```bash
make test          # run the tiny pytest suite
ENABLE_BERTOPIC=0 make run  # use the bundled fixtures; skip optional BERTopic
```

### Full pipeline

```bash
make full          # disables FAST_MODE and triggers live scraping + heavy models
```

### Streamlit UI

```bash
make ui            # launches Streamlit UI for semantic search
```

### CLI search

```bash
python index_search.py --build
python index_search.py --query "waste management" --city "Bengaluru" --k 5
```

## Pipeline overview

1. **Scraping**
   - Reddit via `snscrape` (no API keys).
   - News articles from a small whitelist via `requests` + `BeautifulSoup`.
2. **Preprocess**: text cleaning, spaCy lemmatization, stopword removal, city tagging via regex and NER.
3. **Embeddings**: SBERT (`all-MiniLM-L6-v2`) normalized vectors saved to `models/`.
4. **Topics**: LDA (Gensim) always, BERTopic optional via `ENABLE_BERTOPIC` flag.
5. **NER & Sentiment**: spaCy entity extraction + VADER sentiment scores.
6. **Search**: FAISS cosine index with CLI and Streamlit UI.
7. **Causal (optional)**: DoWhy scaffold showcasing causal inference on toy data.

## Configuration

Adjust settings in `config.py` or set shell environment variables such as `REDDIT_QUERY`, `REDDIT_LIMIT`, `HTTP_PROXY`, and `HTTPS_PROXY`. Use `FAST_MODE=0` or `make full` for live scraping.

## Troubleshooting

- **FAISS/HDBSCAN build issues**: install system deps (`libopenblas-dev`, `build-essential`). In Docker use `apt-get update && apt-get install -y build-essential libopenblas-dev` before pip install.
- **spaCy models**: ensure `python -m spacy download en_core_web_sm`. For transformer model, install `en_core_web_trf` and set `SPACY_MODEL_TRF` in `.env`.
- **Live Reddit errors**: use sample mode for local development; updating the pinned scraper is not a verified fix.
- **SentenceTransformers downloads**: ensure network access; otherwise pre-download models and place in `models/`.

## Testing strategy

- `pytest -q`: runs against sample parquet data (`data/sample`).
- `FAST_MODE=1` (default) skips live scraping, not all model stages. Set `ENABLE_BERTOPIC=0` to explicitly skip BERTopic.

## Data outputs

- Processed corpora: `data/processed/`
- Embeddings: `models/sbert_embeddings.npy`
- FAISS index: `index/sbert.cosine.faiss`

## Scripts

- `scripts/bootstrap.sh`: create venv, install deps, download spaCy model.
- `scripts/run_all.sh`: execute pipeline with `FAST_MODE=1`.
- `scripts/smoke_search.sh`: build FAISS index and run sample queries.

## Make targets

- `make bootstrap`
- `make test`
- `make run`
- `make ui`
- `make full`

