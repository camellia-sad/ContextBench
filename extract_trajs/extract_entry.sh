#!/usr/bin/env bash
# Wrapper: prefer invoking this .sh; logic lives in extract_entry.py (bash).
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/extract_entry.py" "$@"
