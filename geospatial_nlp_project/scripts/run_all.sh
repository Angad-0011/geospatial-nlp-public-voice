#!/usr/bin/env bash
set -euo pipefail

export FAST_MODE=${FAST_MODE:-1}
python run_pipeline.py
