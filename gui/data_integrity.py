"""BrainVision inventory, marker audit, duplicate checks, and blinding support.

Raw files are opened read-only.  The public review manifest contains no source
path, filename-derived participant token, date, or holder configuration.  The
private key is required by the backend to locate a recording, but is never
rendered by the Streamlit review application.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import secrets
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


SUPPORTED_DTYPES = {
    "INT_16": np.dtype("<i2"),
    "UINT_16": np.dtype("<u2"),
    "INT_32": np.dtype("<i4"),
    "IEEE_FLOAT_32": np.dtype("<f4"),
}


@dataclass(frozen=True)
class Marker:
    index: int
    marker_type: str
    description: str
    position: int  # zero based
    size: int
    channel: int
    timestamp: str = ""


@dataclass(frozen=True)
class RecordingInfo:
    header_path: Path
    data_path: Path
    marker_path: Path
    session_name: str
    n_channels: int
    n_samples: int
    sampling_rate_hz: float
    duration_seconds: float
    binary_format: str
    data_orientation: str
    channel_labels: tuple[str, ...]
    channel_resolutions: tuple[float, ...]
    channel_units: tuple[str, ...]
    recording_timestamp: str


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig", errors="replace")


def parse_sections(path: Path) -> dict[str, dict[str, str]]:
    """Parse the INI-like BrainVision format without altering comma fields."""
    sections: dict[str, dict[str, str]] = {}
    current = ""
    for raw_line in read_text(path).splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            sections.setdefault(current, {})
        elif current and "=" in line:
            key, value = line.split("=", 1)
            sections[current][key.strip()] = value.strip()
    return sections


def parse_markers(path: Path) -> list[Marker]:
    rows: list[Marker] = []
    for key, value in parse_sections(path).get("Marker Infos", {}).items():
        if not key.lower().startswith("mk"):
            continue
        fields = value.split(",")
        if len(fields) < 5:
            continue
        try:
            rows.append(
                Marker(
                    index=int(re.sub(r"\D", "", key)),
                    marker_type=fields[0].strip(),
                    description=fields[1].strip(),
                    position=int(fields[2]) - 1,
                    size=int(fields[3]),
                    channel=int(fields[4]),
                    timestamp=fields[5].strip() if len(fields) > 5 else "",
                )
            )
        except ValueError:
            continue
    return sorted(rows, key=lambda row: row.index)


def _channel_fields(sections: dict[str, dict[str, str]], n_channels: int) -> tuple[tuple[str, ...], tuple[float, ...], tuple[str, ...]]:
    labels: list[str] = []
    resolutions: list[float] = []
    units: list[str] = []
    channel_section = sections.get("Channel Infos", {})
    for number in range(1, n_channels + 1):
        fields = channel_section.get(f"Ch{number}", "").split(",")
        labels.append((fields[0] if fields else f"Ch{number}").replace("\\1", ",").strip())
        try:
            resolutions.append(float(fields[2]))
        except (IndexError, ValueError):
            resolutions.append(float("nan"))
        units.append(fields[3].strip() if len(fields) > 3 else "")
    return tuple(labels), tuple(resolutions), tuple(units)


def parse_vhdr(path: Path) -> RecordingInfo:
    sections = parse_sections(path)
    common = sections.get("Common Infos", {})
    binary = sections.get("Binary Infos", {})
    n_channels = int(common.get("NumberOfChannels", "0"))
    interval_us = float(common.get("SamplingInterval", "nan"))
    binary_format = binary.get("BinaryFormat", "").upper()
    if binary_format not in SUPPORTED_DTYPES:
        raise ValueError(f"Unsupported BrainVision BinaryFormat {binary_format!r}: {path}")
    data_path = (path.parent / common.get("DataFile", "")).resolve()
    marker_path = (path.parent / common.get("MarkerFile", "")).resolve()
    if not data_path.is_file() or not marker_path.is_file():
        raise FileNotFoundError(f"Missing linked BrainVision component for {path}")
    n_samples, remainder = divmod(data_path.stat().st_size, n_channels * SUPPORTED_DTYPES[binary_format].itemsize)
    if remainder:
        raise ValueError(f"Raw byte count is not divisible by channel count: {data_path}")
    labels, resolutions, units = _channel_fields(sections, n_channels)
    timestamps = [row.timestamp for row in parse_markers(marker_path) if row.timestamp]
    fs = 1_000_000.0 / interval_us
    return RecordingInfo(
        header_path=path.resolve(),
        data_path=data_path,
        marker_path=marker_path,
        session_name=path.stem,
        n_channels=n_channels,
        n_samples=n_samples,
        sampling_rate_hz=fs,
        duration_seconds=n_samples / fs,
        binary_format=binary_format,
        data_orientation=common.get("DataOrientation", ""),
        channel_labels=labels,
        channel_resolutions=resolutions,
        channel_units=units,
        recording_timestamp=timestamps[0] if timestamps else "",
    )


def discover_recordings(data_root: Path) -> list[RecordingInfo]:
    """Discover continuous BrainVision recordings, excluding stored averages."""
    return [
        parse_vhdr(path)
        for path in sorted(data_root.rglob("*.vhdr"), key=lambda item: str(item).casefold())
        if "Triggers" not in path.stem
    ]


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def marker_sequence_hash(markers: Sequence[Marker], relative: bool = False) -> str:
    if not markers:
        return ""
    origin = markers[0].position if relative else 0
    serial = [
        (row.marker_type, row.description, row.position - origin, row.size, row.channel)
        for row in markers
    ]
    return hashlib.sha256(json.dumps(serial, separators=(",", ":")).encode()).hexdigest()


def checkerboard_events(markers: Sequence[Marker], event_code: str = "S  7") -> np.ndarray:
    normalized = re.sub(r"\s+", "", event_code).upper()
    return np.asarray(
        [
            row.position
            for row in markers
            if row.marker_type.casefold() == "stimulus"
            and re.sub(r"\s+", "", row.description).upper() == normalized
        ],
        dtype=int,
    )


def condition_markers(markers: Sequence[Marker]) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for marker in markers:
        if marker.marker_type.casefold() != "comment":
            continue
        label = marker.description.strip().casefold()
        if label in {"open", "eyes open", "eo"} or "eyes open" in label:
            rows.append(("eyes_open", marker.position))
        elif label in {"close", "closed", "eyes closed", "ec", "ce"} or "eyes close" in label:
            rows.append(("eyes_closed", marker.position))
    return sorted(rows, key=lambda row: row[1])


def condition_intervals(
    markers: Sequence[Marker], n_samples: int, fs: float, maximum_seconds: float = 30.0
) -> dict[str, list[tuple[int, int]]]:
    starts = condition_markers(markers)
    intervals: dict[str, list[tuple[int, int]]] = {"eyes_open": [], "eyes_closed": []}
    cap = int(round(maximum_seconds * fs))
    for index, (condition, start) in enumerate(starts):
        next_start = starts[index + 1][1] if index + 1 < len(starts) else n_samples
        end = min(n_samples, start + cap, next_start)
        if end > start:
            intervals[condition].append((start, end))
    return intervals


def event_blocks(events: Sequence[int], fs: float, gap_seconds: float = 5.0) -> list[np.ndarray]:
    values = np.asarray(events, dtype=int)
    if values.size == 0:
        return []
    split_points = np.flatnonzero(np.diff(values) / fs > gap_seconds) + 1
    return [part for part in np.split(values, split_points) if part.size]


def marker_timing_metrics(events: Sequence[int], fs: float) -> dict[str, object]:
    blocks = event_blocks(events, fs)
    within = np.concatenate([np.diff(block) for block in blocks if block.size > 1]) / fs if blocks else np.array([])
    median_interval = float(np.median(within)) if within.size else float("nan")
    return {
        "event_code": "S  7",
        "event_count": int(len(events)),
        "block_count": len(blocks),
        "events_per_block": "|".join(str(len(block)) for block in blocks),
        "median_iei_seconds": median_interval,
        "minimum_iei_seconds": float(np.min(within)) if within.size else float("nan"),
        "maximum_iei_seconds": float(np.max(within)) if within.size else float("nan"),
        "median_reversal_rate_hz": 1.0 / median_interval if median_interval > 0 else float("nan"),
        "iei_cv": float(np.std(within, ddof=1) / np.mean(within)) if within.size > 1 else float("nan"),
    }


def load_raw_counts(info: RecordingInfo, channels: Sequence[int] | None = None) -> np.ndarray:
    """Return channel x sample raw ADC values using a read-only memmap."""
    if info.data_orientation.upper() != "MULTIPLEXED":
        raise ValueError(f"Only MULTIPLEXED data are supported: {info.header_path}")
    mapped = np.memmap(info.data_path, dtype=SUPPORTED_DTYPES[info.binary_format], mode="r")
    matrix = mapped.reshape(info.n_samples, info.n_channels).T
    if channels is None:
        return matrix
    return matrix[np.asarray(channels, dtype=int)]


def counts_to_microvolts(counts: np.ndarray, info: RecordingInfo, channels: Sequence[int] | None = None) -> np.ndarray:
    selected = np.arange(info.n_channels) if channels is None else np.asarray(channels, dtype=int)
    scale = np.asarray(info.channel_resolutions, dtype=float)[selected]
    units = np.asarray(info.channel_units, dtype=object)[selected]
    normalized_units = [str(unit).strip().replace("µ", "u").replace("μ", "u").casefold() for unit in units]
    if any(unit != "uv" for unit in normalized_units):
        raise ValueError(f"Expected microvolt channel units, found {units.tolist()}")
    return np.asarray(counts, dtype=np.float64) * scale[:, None]


def _random_review_id(existing: set[str]) -> str:
    while True:
        candidate = f"RV-{secrets.token_hex(4).upper()}"
        if candidate not in existing:
            existing.add(candidate)
            return candidate


def build_blinded_manifests(
    data_root: Path,
    public_manifest_path: Path,
    private_key_path: Path,
    overwrite: bool = False,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Create a random, session-level blind and its separately stored key."""
    if (public_manifest_path.exists() or private_key_path.exists()) and not overwrite:
        raise FileExistsError("A review manifest already exists; refusing to silently re-randomize review IDs")
    public_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    private_key_path.parent.mkdir(parents=True, exist_ok=True)
    existing: set[str] = set()
    public_rows: list[dict[str, object]] = []
    private_rows: list[dict[str, object]] = []
    for info in discover_recordings(data_root):
        markers = parse_markers(info.marker_path)
        events = checkerboard_events(markers)
        rests = condition_intervals(markers, info.n_samples, info.sampling_rate_hz)
        protocol_complete = bool(len(events) and len(rests["eyes_open"]) >= 2 and len(rests["eyes_closed"]) >= 2)
        review_id = _random_review_id(existing)
        public_rows.append(
            {
                "review_id": review_id,
                "session": "SESSION_01",
                "n_channels": info.n_channels,
                "sample_count": info.n_samples,
                "duration_seconds": round(info.duration_seconds, 3),
                "sampling_rate_hz": info.sampling_rate_hz,
                "checkerboard_event_count": len(events),
                "eyes_open_segments": len(rests["eyes_open"]),
                "eyes_closed_segments": len(rests["eyes_closed"]),
                "channel_mapping_status": "VERIFIED" if info.n_channels == 11 else "UNRESOLVED",
                "identity_status": "UNRESOLVED",
                "configuration_status": "VERIFIED" if info.n_channels == 11 else "UNRESOLVED",
                "protocol_status": "COMPLETE" if protocol_complete else "INCOMPLETE",
                "manual_review_scope": "IN_SCOPE" if protocol_complete and info.n_channels == 11 else "INTEGRITY_ONLY",
            }
        )
        private_rows.append(
            {
                "review_id": review_id,
                "session": "SESSION_01",
                "source_session_name": info.session_name,
                "source_vhdr": str(info.header_path),
                "source_eeg": str(info.data_path),
                "source_vmrk": str(info.marker_path),
                "recording_timestamp": info.recording_timestamp,
            }
        )
    order = list(range(len(public_rows)))
    secrets.SystemRandom().shuffle(order)
    public_rows = [public_rows[index] for index in order]
    private_rows = [private_rows[index] for index in order]
    _write_csv(public_manifest_path, public_rows)
    _write_csv(private_key_path, private_rows)
    return public_rows, private_rows


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def resolve_review_recording(review_id: str, session: str, private_key_path: Path) -> RecordingInfo:
    matches = [row for row in load_manifest(private_key_path) if row["review_id"] == review_id and row["session"] == session]
    if len(matches) != 1:
        raise KeyError(f"Expected one private mapping for {review_id}/{session}; found {len(matches)}")
    return parse_vhdr(Path(matches[0]["source_vhdr"]))


def integrity_record(info: RecordingInfo) -> dict[str, object]:
    markers = parse_markers(info.marker_path)
    events = checkerboard_events(markers)
    return {
        "file_hash_vhdr": sha256_file(info.header_path),
        "file_hash_eeg": sha256_file(info.data_path),
        "file_hash_vmrk": sha256_file(info.marker_path),
        "sample_count": info.n_samples,
        "duration_seconds": info.duration_seconds,
        "sampling_rate_hz": info.sampling_rate_hz,
        "binary_format": info.binary_format,
        "channel_headers": "|".join(info.channel_labels),
        "channel_units": "|".join(info.channel_units),
        "channel_resolutions": "|".join(str(value) for value in info.channel_resolutions),
        "marker_count": len(markers),
        "marker_sequence_hash": marker_sequence_hash(markers),
        "relative_marker_sequence_hash": marker_sequence_hash(markers, relative=True),
        **marker_timing_metrics(events, info.sampling_rate_hz),
    }


def duplicate_candidates(integrity_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Flag exact data/marker duplicates and metadata-matched near candidates.

    Near candidates remain UNRESOLVED until waveform alignment or acquisition
    records establish identity; they are never pooled automatically.
    """
    candidates: list[dict[str, object]] = []
    for left_index, left in enumerate(integrity_rows):
        for right in integrity_rows[left_index + 1 :]:
            exact_data = left.get("file_hash_eeg") == right.get("file_hash_eeg")
            exact_markers = left.get("marker_sequence_hash") == right.get("marker_sequence_hash")
            near_metadata = (
                left.get("sample_count") == right.get("sample_count")
                and left.get("sampling_rate_hz") == right.get("sampling_rate_hz")
                and left.get("channel_headers") == right.get("channel_headers")
            )
            if exact_data or exact_markers or near_metadata:
                candidates.append(
                    {
                        "left_review_id": left.get("review_id", ""),
                        "right_review_id": right.get("review_id", ""),
                        "exact_data": exact_data,
                        "exact_marker_sequence": exact_markers,
                        "metadata_near_candidate": near_metadata,
                        "automated_status": "FAIL" if exact_data else "WARNING",
                        "identity_status": "UNRESOLVED",
                    }
                )
    return candidates


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def export_integrity_inventory(data_root: Path, path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for info in discover_recordings(data_root):
        row = {"source_session_name": info.session_name, **integrity_record(info)}
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(path, rows)
    return rows


def channel_map(n_channels: int) -> list[dict[str, object]]:
    if n_channels != 11:
        return [{"status": "UNRESOLVED", "detail": "No independently documented derivation map for this channel count"}]
    rows = []
    for site, (teeg, eeg) in enumerate(((1, 2), (3, 4), (5, 6), (7, 8)), start=1):
        rows.append({"site": str(site), "tEEG_channel": f"Ch{teeg}", "eEEG_channel": f"Ch{eeg}", "interface": "blinded"})
    rows.extend(
        [
            {"site": "reference TCRE", "tEEG_channel": "Ch9", "eEEG_channel": "Ch10", "interface": "reference"},
            {"site": "reference disc", "tEEG_channel": "", "eEEG_channel": "Ch11", "interface": "reference"},
        ]
    )
    return rows


def dataclass_to_json(value: RecordingInfo) -> str:
    row = asdict(value)
    for key in ("header_path", "data_path", "marker_path"):
        row[key] = str(row[key])
    return json.dumps(row, indent=2)


def group_exact_hashes(rows: Iterable[dict[str, object]], field: str) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        value = str(row.get(field, ""))
        if value:
            grouped[value].append(str(row.get("review_id", row.get("source_session_name", ""))))
    return {key: values for key, values in grouped.items() if len(values) > 1}
