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
    claim_segment,
    finish_segment,
    pack_project,
    parse_env_pairs,
    paused_time,
    read_segments,
    sanitize_output_subdir,
    script_function_name,
    script_paths,
    substitute_output_dir,
    tail,
    unpack_project,
    write_segment,
)


class FakeClaims(dict):
    """modal.Dict stand-in: put(..., skip_if_exists=True) refuses an existing key."""

    def put(self, key, value, *, skip_if_exists=False):
        if skip_if_exists and key in self:
            return False
        self[key] = value
        return True


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


class TestScriptSubmission:
    def test_script_paths_adds_uncovered_script(self):
        assert script_paths("scripts/lib, configs", "drive.py") == ["scripts/lib", "configs", "drive.py"]

    def test_script_paths_keeps_covered_script(self):
        assert script_paths(["scripts/", "configs"], "scripts/drive.py") == ["scripts/", "configs"]
        assert script_paths("scripts/drive.py", "scripts/drive.py") == ["scripts/drive.py"]

    def test_script_paths_prefix_is_not_a_folder_match(self):
        assert script_paths("scr", "scripts/drive.py") == ["scr", "scripts/drive.py"]

    def test_function_names(self):
        assert script_function_name(True) == "run_script_gpu"
        assert script_function_name(False) == "run_script_cpu"


class TestSubmitCLI:
    """The script/call subcommands of scripts/modal_submit.py, with Modal stubbed out."""

    @pytest.fixture
    def submit(self, monkeypatch):
        import modal_submit
        monkeypatch.setattr(modal_submit, "check_modal_installed", lambda: None)
        return modal_submit

    def test_script_command(self, submit, monkeypatch):
        seen = {}
        monkeypatch.setattr(submit, "spawn_script", lambda *a, **k: seen.update(args=a, kwargs=k))
        monkeypatch.setattr(sys, "argv", ["modal_submit.py", "script", "--project-dir", "../p", "--include", "s,c",
                                          "--script", "s/d.py", "--args", "c/r.yaml --out {output_dir}",
                                          "--output-subdir", "run1", "--env", "A=1", "--cpu"])
        submit.main()
        assert seen["args"] == ("../p", "s,c", "s/d.py")
        assert seen["kwargs"]["args"] == "c/r.yaml --out {output_dir}"
        assert seen["kwargs"]["output_subdir"] == "run1"
        assert seen["kwargs"]["use_gpu"] is False and seen["kwargs"]["env"] == "A=1"

    def test_call_command(self, submit, monkeypatch):
        seen = {}
        monkeypatch.setattr(submit, "call_status", lambda *a, **k: seen.update(args=a, kwargs=k))
        monkeypatch.setattr(sys, "argv", ["modal_submit.py", "call", "fc-123", "--wait", "--interval", "5"])
        submit.main()
        assert seen["args"] == ("fc-123",) and seen["kwargs"] == {"wait": True, "interval": 5.0}

    def test_script_command_max_segments(self, submit, monkeypatch):
        seen = {}
        monkeypatch.setattr(submit, "spawn_script", lambda *a, **k: seen.update(kwargs=k))
        monkeypatch.setattr(sys, "argv", ["modal_submit.py", "script", "--project-dir", ".", "--include", "s",
                                          "--script", "s/d.py", "--max-segments", "20"])
        submit.main()
        assert seen["kwargs"]["max_segments"] == 20


def _seg(k, call_id, outcome, **extra):
    return {"segment": k, "call_id": call_id, "outcome": outcome, **extra}


class TestChainClaims:
    """claim_segment: which call runs which segment of a server-side chain."""

    def claim(self, segments, requested, call_id, live=(), claims=None, max_segments=5):
        claims = FakeClaims() if claims is None else claims
        return claim_segment(segments, requested, call_id, max_segments, claims, "vol/run",
                             is_live=lambda c: c in live)

    def test_first_submission_runs_segment_0(self):
        assert self.claim([], None, "fc-a") == (0, "claimed")

    def test_successor_runs_its_segment(self):
        assert self.claim([_seg(0, "fc-a", "paused")], 1, "fc-b") == (1, "claimed")

    def test_duplicate_successor_is_refused(self):
        claims = FakeClaims()
        segs = [_seg(0, "fc-a", "paused")]
        assert self.claim(segs, 1, "fc-b", claims=claims)[0] == 1
        k, reason = self.claim(segs, 1, "fc-c", claims=claims, live={"fc-b"})
        assert k is None and "held by call fc-b" in reason
        # refused even when the holder has died: only a submission skips a dead segment
        assert self.claim(segs, 1, "fc-c", claims=claims)[0] is None

    def test_second_submission_while_chain_runs_is_refused(self):
        k, reason = self.claim([_seg(0, "fc-a", "running")], None, "fc-b", live={"fc-a"})
        assert k is None and "fc-a is still running" in reason

    def test_submission_while_successor_is_pending_is_refused(self):
        segs = [_seg(0, "fc-a", "paused", next_call_id="fc-b")]
        assert self.claim(segs, None, "fc-c", live={"fc-b"})[0] is None

    def test_submission_after_a_dead_segment_continues_after_it(self):
        assert self.claim([_seg(0, "fc-a", "running")], None, "fc-b") == (1, "claimed")

    def test_submission_skips_a_segment_claimed_by_a_dead_call(self):
        claims = FakeClaims({"vol/run#1": "fc-dead"})
        assert self.claim([_seg(0, "fc-a", "failed")], None, "fc-b", claims=claims) == (2, "claimed")

    def test_submission_does_not_skip_a_segment_claimed_by_a_live_call(self):
        claims = FakeClaims({"vol/run#1": "fc-live"})
        k, _ = self.claim([_seg(0, "fc-a", "failed")], None, "fc-b", claims=claims, live={"fc-live"})
        assert k is None

    def test_restarted_call_reruns_its_unfinished_segment(self):
        claims = FakeClaims({"vol/run#1": "fc-b"})
        segs = [_seg(0, "fc-a", "paused"), _seg(1, "fc-b", "running")]
        assert self.claim(segs, 1, "fc-b", claims=claims) == (1, "claimed")

    def test_restarted_call_does_not_rerun_an_ended_segment(self):
        segs = [_seg(0, "fc-a", "paused", next_call_id="fc-b")]
        k, reason = self.claim(segs, None, "fc-a")
        assert k is None and "already ran segment 0" in reason

    @pytest.mark.parametrize("requested", [None, 2])
    def test_finished_run_never_runs_again(self, requested):
        segs = [_seg(0, "fc-a", "paused"), _seg(1, "fc-b", "finished")]
        assert self.claim(segs, requested, "fc-c") == (None, "the run has finished")

    def test_cap(self):
        k, reason = self.claim([_seg(k, f"fc-{k}", "paused") for k in range(5)], 5, "fc-x")
        assert k is None and "max_segments = 5" in reason


class TestChainFinish:
    def test_finished(self):
        rec = _seg(0, "fc-a", "running")
        assert finish_segment(rec, 0, "done\n", [], 5) is False
        assert rec["outcome"] == "finished" and rec["returncode"] == 0

    def test_paused_continues(self):
        rec = _seg(1, "fc-b", "running")
        prev = [_seg(0, "fc-a", "paused", t=1.0)]
        assert finish_segment(rec, 3, "paused at t = 2.5000, step 9\n", prev, 5) is True
        assert rec["outcome"] == "paused" and rec["t"] == 2.5 and "stop_reason" not in rec

    def test_paused_without_progress_stops(self):
        rec = _seg(1, "fc-b", "running")
        prev = [_seg(0, "fc-a", "paused", t=2.5)]
        assert finish_segment(rec, 3, "paused at t = 2.5, step 9\n", prev, 5) is False
        assert rec["stop_reason"].startswith("no progress")

    def test_paused_at_the_cap_stops(self):
        rec = _seg(4, "fc-e", "running")
        assert finish_segment(rec, 3, "", [], 5) is False
        assert "max_segments = 5" in rec["stop_reason"]

    def test_crash_stops(self):
        rec = _seg(0, "fc-a", "running")
        assert finish_segment(rec, 1, "Traceback ...\n", [], 5) is False
        assert rec["outcome"] == "failed"

    def test_paused_time(self):
        assert paused_time("x\npaused at t = 1.5, step 3\npaused at t = 2e+01, step 4\n") == 20.0
        assert paused_time("[run] paused at t = 12.3456, step 7, after 23 h") == 12.3456
        assert paused_time("no pause here") is None

    def test_segment_records_round_trip(self, tmp_path):
        write_segment(tmp_path, _seg(1, "fc-b", "paused"))
        write_segment(tmp_path, _seg(0, "fc-a", "paused"))
        write_segment(tmp_path, _seg(1, "fc-b", "finished"))
        assert [(s["segment"], s["outcome"]) for s in read_segments(tmp_path)] == [(0, "paused"), (1, "finished")]
        assert read_segments(tmp_path / "missing") == []
        assert not list((tmp_path / "chain").glob("*.tmp"))
