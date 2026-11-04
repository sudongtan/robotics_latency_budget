#!/usr/bin/env bash
# Run MuJoCo's `mjpython` (needed for the native viewer on macOS) inside the uv env.
#
# uv's standalone Python links libpython3.12.dylib via @rpath, but mjpython only searches
# the venv for it and fails with "Library not loaded: @rpath/libpython3.12.dylib".
# Point the loader at the interpreter's LIBDIR so it is found.
#
# Usage: scripts/mjpython.sh scripts/view_env.py [args...]
set -euo pipefail

libdir="$(uv run python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"
export DYLD_FALLBACK_LIBRARY_PATH="${libdir}${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"
exec uv run mjpython "$@"
