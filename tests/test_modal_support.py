"""
Tests for scripts/modal_support.py, the Modal-free helpers behind modal_app.run_script_*.
"""

import io
import sys
import tarfile
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from modal_support import (  # noqa: E402
    PeriodicCommitter,
    pack_project,
    parse_env_pairs,
    sanitize_output_subdir,
    substitute_output_dir,
    tail,
    unpack_project,
)


def _make_project(root: Path) -> None:
    (root / "pkg" / "sub").mkdir(parents=True)
    (root / "pkg" / "run.py").write_text("print('hi')\n")
    (root / "pkg" / "sub" / "lib.py").write_text("X = 1\n")
    (root / "pkg" / "__pycache__").mkdir()
    (root / "pkg" / "__pycache__" / "run.cpython-311.pyc").write_bytes(b"\0")
    (root / "configs").mkdir()
    (root / "configs" / "a.yaml").write_text("a: 1\n")
    (root / "data").mkdir()
    (root / "data" / "big.bin").write_bytes(b"\0" * 1000)


class TestPackUnpack:
    def test_round_trip_keeps_layout_and_skips_pycache(self, tmp_path):
        src = tmp_path / "src"
        _make_project(src)
        data = pack_project(src, ["pkg", "configs/a.yaml"])
        names = unpack_project(data, tmp_path / "dest")
        assert sorted(names) == ["configs/a.yaml", "pkg/run.py", "pkg/sub/lib.py"]
        assert (tmp_path / "dest" / "pkg" / "sub" / "lib.py").read_text() == "X = 1\n"
        assert not (tmp_path / "dest" / "data").exists()

    def test_duplicate_includes_are_packed_once(self, tmp_path):
        _make_project(tmp_path)
        data = pack_project(tmp_path, ["pkg", "pkg/run.py"])
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            names = [m.name for m in tar.getmembers()]
        assert names.count("pkg/run.py") == 1

    @pytest.mark.parametrize("bad", ["../outside", "/etc/passwd", ""])
    def test_refuses_paths_outside_root(self, tmp_path, bad):
        _make_project(tmp_path / "p")
        with pytest.raises(ValueError):
            pack_project(tmp_path / "p", [bad])

    def test_refuses_missing_path(self, tmp_path):
        _make_project(tmp_path)
        with pytest.raises(ValueError):
            pack_project(tmp_path, ["nope"])

    def test_refuses_symlinks(self, tmp_path):
        _make_project(tmp_path)
        (tmp_path / "pkg" / "link.py").symlink_to(tmp_path / "pkg" / "run.py")
        with pytest.raises(ValueError):
            pack_project(tmp_path, ["pkg"])

    def test_size_limit(self, tmp_path):
        _make_project(tmp_path)
        with pytest.raises(ValueError):
            pack_project(tmp_path, ["data"], max_bytes=10)

    @pytest.mark.parametrize("name", ["../evil.py", "/abs.py", "a/../../evil.py"])
    def test_unpack_refuses_unsafe_members(self, tmp_path, name):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            info = tarfile.TarInfo(name)
            payload = b"x"
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        with pytest.raises(ValueError):
            unpack_project(buf.getvalue(), tmp_path / "dest")
        assert not (tmp_path / "evil.py").exists()

    def test_unpack_refuses_links(self, tmp_path):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            info = tarfile.TarInfo("link")
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            tar.addfile(info)
        with pytest.raises(ValueError):
            unpack_project(buf.getvalue(), tmp_path / "dest")


class TestArguments:
    def test_substitute_output_dir(self):
        args = ["cfg.yaml", "--base", "{output_dir}", "--log={output_dir}/log.txt"]
        assert substitute_output_dir(args, "/results/run1") == [
            "cfg.yaml", "--base", "/results/run1", "--log=/results/run1/log.txt"]

    @pytest.mark.parametrize("name,expected", [("run1", "run1"), ("/a/b/", "a/b"), ("x_1.2-y", "x_1.2-y")])
    def test_sanitize_ok(self, name, expected):
        assert sanitize_output_subdir(name) == expected

    @pytest.mark.parametrize("name", ["", "/", "a/../b", "a b", "a/./b", "a;rm"])
    def test_sanitize_rejects(self, name):
        with pytest.raises(ValueError):
            sanitize_output_subdir(name)

    def test_parse_env_pairs(self):
        assert parse_env_pairs("") == {}
        assert parse_env_pairs("JAX_PLATFORMS=cuda, X_1=a=b,") == {"JAX_PLATFORMS": "cuda", "X_1": "a=b"}

    @pytest.mark.parametrize("text", ["NOVALUE", "1BAD=x", "A-B=c", "=x"])
    def test_parse_env_pairs_rejects(self, text):
        with pytest.raises(ValueError):
            parse_env_pairs(text)

    def test_tail(self):
        text = "\n".join(str(i) for i in range(100))
        assert tail(text, 3) == "97\n98\n99"


class TestPeriodicCommitter:
    def test_commits_repeatedly_and_stops(self):
        calls = []
        with PeriodicCommitter(lambda: calls.append(time.time()), interval=0.02) as pc:
            time.sleep(0.15)
        n = len(calls)
        time.sleep(0.06)
        assert n >= 3
        assert len(calls) == n          # no commits after stop
        assert pc.n_commits == n

    def test_errors_do_not_stop_the_thread(self):
        errors = []
        count = {"n": 0}

        def flaky():
            count["n"] += 1
            if count["n"] % 2:
                raise RuntimeError("transient")

        pc = PeriodicCommitter(flaky, interval=0.02, on_error=errors.append).start()
        time.sleep(0.15)
        pc.stop()
        assert len(errors) >= 2 and pc.n_commits >= 2
        assert all(isinstance(e, RuntimeError) for e in errors)

    def test_rejects_non_positive_interval(self):
        with pytest.raises(ValueError):
            PeriodicCommitter(lambda: None, interval=0)

    def test_thread_is_daemon(self):
        pc = PeriodicCommitter(lambda: None, interval=10)
        assert isinstance(pc._thread, threading.Thread) and pc._thread.daemon
