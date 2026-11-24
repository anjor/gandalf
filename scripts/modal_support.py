"""
Helpers for running external driver scripts on Modal (modal_app.py, run_script_*).

This module does not import Modal, so it can be tested without it. It covers:

- packing selected files of a caller's project into a gzipped tarball, and
  unpacking it safely inside the container;
- substituting the output folder into the driver's arguments;
- sanitising the output subdirectory name;
- committing the results volume periodically from a background thread, so that
  restart files written by a long run survive a timeout or a preemption;
- the bookkeeping of server-side chains (a paused run spawns its own successor).
"""

from __future__ import annotations

import io
import json
import os
import re
import tarfile
import threading
import time
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Optional, Protocol

# Default upper limit on the packed project (bytes); Modal accepts larger inputs,
# but a project tarball beyond this size usually means data was included by mistake.
DEFAULT_MAX_PROJECT_BYTES = 200 * 1024 * 1024

OUTPUT_DIR_PLACEHOLDER = "{output_dir}"

_SKIP_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints"}


def _relative_inside(root: Path, rel: str) -> Path:
    """Resolve `rel` under `root` and refuse paths that leave it."""
    if not rel or PurePosixPath(rel).is_absolute():
        raise ValueError(f"include paths must be relative to the project root, got {rel!r}")
    path = (root / rel).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"include path {rel!r} leaves the project root {root}")
    return path


def pack_project(root: str | Path, include: Iterable[str],
                 max_bytes: int = DEFAULT_MAX_PROJECT_BYTES) -> bytes:
    """Gzipped tarball of the files and folders `include` (paths relative to `root`).

    Folders are added recursively, without __pycache__, .git and notebook checkpoints;
    symbolic links are refused. Member names are the paths relative to `root`, so the
    archive unpacks into the same layout. Raises ValueError if a path does not exist,
    leaves `root`, or the archive exceeds `max_bytes`.
    """
    root = Path(root).resolve()
    paths = []
    for rel in include:
        path = _relative_inside(root, rel)
        if not path.exists():
            raise ValueError(f"include path {rel!r} does not exist under {root}")
        paths.append(path)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        seen: set[Path] = set()
        for path in paths:
            files = [path] if path.is_file() else sorted(
                p for p in path.rglob("*")
                if p.is_file() and not (set(p.relative_to(root).parts) & _SKIP_DIRS))
            for f in files:
                if f.is_symlink():
                    raise ValueError(f"refusing to pack symbolic link {f}")
                if f in seen:
                    continue
                seen.add(f)
                tar.add(f, arcname=f.relative_to(root).as_posix(), recursive=False)
    data = buf.getvalue()
    if len(data) > max_bytes:
        raise ValueError(f"packed project is {len(data)} bytes, more than max_bytes = {max_bytes}")
    return data


def unpack_project(data: bytes, dest: str | Path) -> list[str]:
    """Unpack a tarball made by pack_project into `dest`; returns the member names.

    Only regular files and folders are extracted, and members with absolute names or
    names containing '..' are refused.
    """
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    names = []
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        members = tar.getmembers()
        for m in members:
            p = PurePosixPath(m.name)
            if p.is_absolute() or ".." in p.parts:
                raise ValueError(f"refusing archive member {m.name!r}")
            if not (m.isfile() or m.isdir()):
                raise ValueError(f"refusing archive member {m.name!r} of type {m.type!r}")
        for m in members:
            target = dest / m.name
            if m.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(m)
            assert src is not None
            with open(target, "wb") as fh:
                fh.write(src.read())
            names.append(m.name)
    return names


def sanitize_output_subdir(name: str) -> str:
    """A relative folder name for the results volume: letters, digits, '.', '_', '-' and '/'
    separators, no empty, '.' or '..' components."""
    parts = [p for p in name.strip().strip("/").split("/") if p]
    if not parts:
        raise ValueError("output_subdir must not be empty")
    for p in parts:
        if p in (".", "..") or not re.fullmatch(r"[A-Za-z0-9._-]+", p):
            raise ValueError(f"invalid output_subdir component {p!r}")
    return "/".join(parts)


def substitute_output_dir(args: Iterable[str], output_dir: str) -> list[str]:
    """Replace every occurrence of '{output_dir}' in the driver's arguments."""
    return [a.replace(OUTPUT_DIR_PLACEHOLDER, output_dir) for a in args]


APP_NAME = "gandalf-krmhd"


def script_paths(include: str | Iterable[str], script: str) -> list[str]:
    """The paths to pack: `include` (comma-separated string or list), plus `script` if no entry covers it."""
    items = include.split(",") if isinstance(include, str) else list(include)
    paths = [p.strip() for p in items if p.strip()]
    if not any(script == p or script.startswith(p.rstrip("/") + "/") for p in paths):
        paths.append(script)
    return paths


def script_function_name(use_gpu: bool) -> str:
    """Name of the deployed function that runs an external driver script."""
    return "run_script_gpu" if use_gpu else "run_script_cpu"


def parse_env_pairs(text: str) -> dict[str, str]:
    """Parse "KEY=VALUE,KEY2=VALUE2" into a dict; keys must be valid environment names."""
    out: dict[str, str] = {}
    for item in (p.strip() for p in text.split(",")):
        if not item:
            continue
        key, sep, value = item.partition("=")
        key = key.strip()
        if not sep or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"invalid environment assignment {item!r}")
        out[key] = value
    return out


class PeriodicCommitter:
    """Call `commit()` every `interval` seconds from a daemon thread until stopped.

    Errors from `commit` are passed to `on_error` (default: ignored) and do not stop
    the thread; `stop()` performs no final commit, which the caller does itself.
    """

    def __init__(self, commit: Callable[[], object], interval: float,
                 on_error: Optional[Callable[[BaseException], None]] = None) -> None:
        if interval <= 0:
            raise ValueError("interval must be positive")
        self._commit = commit
        self._interval = float(interval)
        self._on_error = on_error
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.n_commits = 0

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self._commit()
                self.n_commits += 1
            except BaseException as exc:  # noqa: BLE001 - reported, never fatal
                if self._on_error is not None:
                    self._on_error(exc)

    def start(self) -> "PeriodicCommitter":
        self._thread.start()
        return self

    def stop(self, timeout: Optional[float] = None) -> None:
        self._stop.set()
        self._thread.join(timeout)

    def __enter__(self) -> "PeriodicCommitter":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()


def tail(text: str, n_lines: int = 40) -> str:
    """The last `n_lines` lines of `text`."""
    lines = text.splitlines()
    return "\n".join(lines[-n_lines:])


# =============================================================================
# Server-side chains (modal_app.run_script_*, max_segments > 1)
# =============================================================================
#
# A chain is one run folder on the results volume and a sequence of calls, the
# segments, that each run the same driver command. A driver that exits with
# PAUSED_EXIT has saved its state and continues with the same command; the call
# then spawns the next segment itself. Each segment k writes only its own record,
# <run folder>/chain/segment_<k>.json, so no two containers write the same file.
# Which call runs segment k is decided by an atomic claim in a modal.Dict.

PAUSED_EXIT = 3
CHAIN_DIR = "chain"
_PAUSED_AT = re.compile(r"paused at t = ([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)")


class Claims(Protocol):
    """The part of modal.Dict that chains use; put(..., skip_if_exists=True) is atomic."""

    def put(self, key: str, value: str, *, skip_if_exists: bool = False) -> bool: ...

    def get(self, key: str) -> Any: ...


def paused_time(log: str) -> Optional[float]:
    """Simulation time of the last 'paused at t = <x>' line in a driver's output, if any."""
    found = _PAUSED_AT.findall(log)
    return float(found[-1]) if found else None


def read_segments(run_dir: str | Path) -> list[dict]:
    """The segment records of the chain in `run_dir`, in segment order."""
    records = [json.loads(p.read_text()) for p in (Path(run_dir) / CHAIN_DIR).glob("segment_*.json")]
    return sorted(records, key=lambda r: r["segment"])


def write_segment(run_dir: str | Path, record: dict) -> None:
    """Write a segment record atomically (temporary file, then rename)."""
    folder = Path(run_dir) / CHAIN_DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"segment_{record['segment']:04d}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=1))
    os.replace(tmp, path)


def claim_segment(segments: list[dict], requested: Optional[int], call_id: str, max_segments: int,
                  claims: Claims, run_key: str, is_live: Callable[[str], bool]) -> tuple[Optional[int], str]:
    """The segment this call runs, or (None, reason) when it must not run.

    `segments` are the chain's records, `requested` the segment a predecessor spawned this
    call for (None for a submission from outside), and `is_live(call_id)` tells whether a
    call may still be pending or running. A call Modal restarts (preemption) has the same
    call ID and runs its segment again unless that segment already ended. A finished run
    never runs again, and no segment number reaches `max_segments`.
    """
    mine = [s for s in segments if s["call_id"] == call_id]
    if mine:
        if mine[-1]["outcome"] != "running":
            return None, f"this call already ran segment {mine[-1]['segment']}"
        k = mine[-1]["segment"]
    elif any(s["outcome"] == "finished" for s in segments):
        return None, "the run has finished"
    elif requested is not None:
        k = requested
    else:
        if segments:
            last = segments[-1]
            live = last["call_id"] if last["outcome"] == "running" else last.get("next_call_id")
            if live and is_live(live):
                return None, f"call {live} is still running this chain"
        k = segments[-1]["segment"] + 1 if segments else 0
    while k < max_segments:
        key = f"{run_key}#{k}"
        if claims.put(key, call_id, skip_if_exists=True):
            return k, "claimed"
        holder = claims.get(key)
        if holder == call_id:
            return k, "claimed"
        if requested is not None or mine or (holder is not None and is_live(holder)):
            return None, f"segment {k} is held by call {holder}"
        k += 1  # a submission from outside skips a segment whose call died before recording it
    return None, f"max_segments = {max_segments} reached"


def finish_segment(record: dict, returncode: int, log: str, previous: list[dict],
                   max_segments: int) -> bool:
    """Fill in a segment's outcome; True if the chain continues with a new segment.

    A paused segment continues the chain unless the next segment would reach
    `max_segments` or its 'paused at t = ...' time did not advance past the previous
    paused segment's. Any exit status other than 0 and PAUSED_EXIT stops the chain.
    """
    outcome = {0: "finished", PAUSED_EXIT: "paused"}.get(returncode, "failed")
    t = paused_time(log) if outcome == "paused" else None
    record.update(outcome=outcome, returncode=returncode, t=t, ended=time.strftime("%Y-%m-%d %H:%M:%S"))
    if outcome != "paused":
        return False
    earlier = [s["t"] for s in previous
               if s["segment"] < record["segment"] and s["outcome"] == "paused" and s.get("t") is not None]
    if t is not None and earlier and t <= earlier[-1]:
        record["stop_reason"] = f"no progress: paused at t = {t}, previous segment at t = {earlier[-1]}"
    elif record["segment"] + 1 >= max_segments:
        record["stop_reason"] = f"max_segments = {max_segments} reached"
    return "stop_reason" not in record
