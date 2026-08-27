"""Launch MNE's native interactive raw browser for a blinded review record."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .data_integrity import load_manifest, resolve_review_recording
    from .mne_backend import load_brainvision, native_browser
except ImportError:
    from data_integrity import load_manifest, resolve_review_recording
    from mne_backend import load_brainvision, native_browser


GUI_DIR = Path(__file__).resolve().parent
PUBLIC = GUI_DIR / "review_data" / "blinded_review_manifest.csv"
PRIVATE = GUI_DIR / "private" / "blinding_key.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-id", required=True)
    parser.add_argument("--session", default="SESSION_01")
    parser.add_argument("--site", type=int, choices=(1, 2, 3, 4))
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=20.0)
    args = parser.parse_args()
    public_ids = {row["review_id"] for row in load_manifest(PUBLIC)}
    if args.review_id not in public_ids:
        raise SystemExit(f"Unknown blinded review ID: {args.review_id}")
    info = resolve_review_recording(args.review_id, args.session, PRIVATE)
    recording = load_brainvision(info.header_path, preload=False)
    picks = None if args.site is None else [(args.site - 1) * 2, (args.site - 1) * 2 + 1]
    native_browser(recording, picks=picks, start=args.start, duration=args.duration, block=True)


if __name__ == "__main__":
    main()
