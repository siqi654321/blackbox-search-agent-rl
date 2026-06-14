#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OVERLAY_DIR="${1:-$ROOT/overlays/prorl-agent-server}"
TARGET_DIR="${2:-$ROOT/submodules/prorl-agent-server}"
VERL_DIR="${VERL_DIR:-$ROOT/submodules/verl}"

if [[ ! -d "$OVERLAY_DIR" ]]; then
  echo "Overlay directory not found: $OVERLAY_DIR" >&2
  exit 1
fi
if [[ ! -d "$TARGET_DIR/.git" && ! -f "$TARGET_DIR/.git" ]]; then
  echo "ProRL-Agent-Server submodule checkout not found: $TARGET_DIR" >&2
  echo "Run: git submodule update --init --recursive" >&2
  exit 1
fi
if [[ ! -d "$VERL_DIR/.git" && ! -f "$VERL_DIR/.git" ]]; then
  echo "verl submodule checkout not found: $VERL_DIR" >&2
  echo "Run: git submodule update --init --recursive" >&2
  exit 1
fi

rsync -a "$OVERLAY_DIR/" "$TARGET_DIR/"


echo "Applied overlay: $OVERLAY_DIR -> $TARGET_DIR"
echo "Use VERL_ROOT=$VERL_DIR when launching from the ProRL-Agent-Server checkout."
echo "Submodule status after overlay:"
git -C "$TARGET_DIR" status --short
