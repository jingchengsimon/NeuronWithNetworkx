"""Backward-compatible entrypoint for legacy segment-center script name."""

from __future__ import annotations

try:
    from .generate_segment_coordinates import main
except ImportError:
    # Allow direct execution: python analysis/morphology/generate_segment_center_coordinates.py
    from generate_segment_coordinates import main


if __name__ == "__main__":
    main()
