#!/bin/sh
# Runs each service's test suite in a separate pytest process:
# index/ and search/ both have top-level config.py / schemas.py modules,
# so their tests cannot share one interpreter.
set -eu

cd "$(dirname "$0")/.."

PYTEST="${PYTEST:-.venv/bin/pytest}"
if [ ! -x "$PYTEST" ]; then
  PYTEST="pytest"
fi

"$PYTEST" tests/index_service "$@"
"$PYTEST" tests/search_service "$@"
"$PYTEST" tests/eval_service "$@"
