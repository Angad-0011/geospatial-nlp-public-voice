.PHONY: bootstrap test run ui full

bootstrap:
bash scripts/bootstrap.sh

test:
pytest -q

run:
FAST_MODE=1 python run_pipeline.py

ui:
streamlit run app_streamlit.py

full:
FAST_MODE=0 python run_pipeline.py
