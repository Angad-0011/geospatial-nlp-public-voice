# geospatial_nlp_project

City-wise public discourse analysis for Reddit and news content with NLP pipelines, topic modeling, and search.

## Quickstart

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
make run           # fast pipeline using sample data
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

Adjust settings in `config.py`. Environment variables (see `.env.example`) override defaults for proxies, scraping, and enabling full runs.

## Troubleshooting

- **FAISS/HDBSCAN build issues**: install system deps (`libopenblas-dev`, `build-essential`). In Docker use `apt-get update && apt-get install -y build-essential libopenblas-dev` before pip install.
- **spaCy models**: ensure `python -m spacy download en_core_web_sm`. For transformer model, install `en_core_web_trf` and set `SPACY_MODEL_TRF` in `.env`.
- **snscrape errors**: upgrade with `pip install --upgrade snscrape`.
- **SentenceTransformers downloads**: ensure network access; otherwise pre-download models and place in `models/`.

## Testing strategy

- `pytest -q`: runs against sample parquet data (`data/sample`).
- Heavy steps (BERTopic, live scraping) are skipped when `FAST_MODE=1` (default). Pass `FAST_MODE=0` or `--full` to enable.

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

