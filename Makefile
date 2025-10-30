.PHONY: bootstrap test run ui full

bootstrap:
	@echo "🔧 Setting up environment..."
	python -m venv .venv
	. .venv/bin/activate && pip install -U pip wheel
	. .venv/bin/activate && pip install -r requirements.txt
	python -m spacy download en_core_web_sm

test:
	@echo "🧪 Running test suite..."
	pytest -q

run:
	@echo "⚡ Running fast pipeline on sample data..."
	FAST_MODE=1 python run_pipeline.py

ui:
	@echo "🌐 Launching Streamlit UI..."
	streamlit run app_streamlit.py

full:
	@echo "🚀 Running full pipeline (may take time)..."
	FAST_MODE=0 python run_pipeline.py
