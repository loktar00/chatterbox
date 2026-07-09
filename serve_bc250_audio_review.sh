#!/usr/bin/env bash
set -euo pipefail

cd /root/chatterbox

# Rebuilds the ignored listen-test page from existing WAV/JSON artifacts and
# serves the repo root so relative audio links resolve from a remote browser.
# This does not load Chatterbox, generate audio, start a TTS worker, or touch
# ROCm/HIP.
./prepare_bc250_audio_review.py

exec python3 -m http.server "${PORT:-8020}" \
  --bind "${HOST:-0.0.0.0}" \
  --directory /root/chatterbox
