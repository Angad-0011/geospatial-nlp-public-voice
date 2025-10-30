#!/usr/bin/env bash
set -euo pipefail

python index_search.py --build
python index_search.py --query "urban mobility" --city "Delhi" --k 5
