"""
Helpers for running external driver scripts on Modal (modal_app.py, run_script_*).

This module does not import Modal, so it can be tested without it. It covers:

- packing selected files of a caller's project into a gzipped tarball, and
  unpacking it safely inside the container;
- substituting the output folder into the driver's arguments;
- sanitising the output subdirectory name;
- committing the results volume periodically from a background thread, so that
  restart files written by a long run survive a timeout or a preemption.
"""

from __future__ import annotations

import io
import re
import tarfile
import threading
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, Optional

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
