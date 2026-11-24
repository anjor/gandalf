"""
Server-side chains in modal_app.run_script_* (Issue #190), with Modal replaced by a stub.

Each test imports modal_app against a stub `modal` module, runs a real driver script in
a subprocess, and executes the calls the segments spawn, as Modal would.
"""

import importlib
import io
import itertools
import sys
import tarfile
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

# A driver that follows the chain contract: each run advances by one unit of "time",
# pauses (exit 3, 'paused at t = ...') until `total` units are done, then exits 0.
# Modes: "crash" exits 1 at the second run, "stuck" pauses without advancing.
DRIVER = '''
import sys
from pathlib import Path
out, total, mode = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
done = int((out / "progress").read_text()) if (out / "progress").exists() else 0
if mode == "crash" and done == 1:
    sys.exit(1)
if mode != "stuck":
    done += 1
(out / "progress").write_text(str(done))
if done < total:
    print(f"paused at t = {float(done)}, step {done}")
    sys.exit(3)
print("finished")
'''


class _Dict(dict):
    def put(self, key, value, *, skip_if_exists=False):
        if skip_if_exists and key in self:
            return False
        self[key] = value
        return True


class _Chainable:
    """Stands for Modal's builder objects (Image, App.function options): every call returns self."""

    def __getattr__(self, name):
        return lambda *a, **k: self


class StubModal(types.ModuleType):
    """The parts of the modal package that modal_app uses."""

    def __init__(self):
        super().__init__("modal")
        self.spawned = []          # (call_id, function, args, kwargs) not yet run
        self.call_id = None        # the call the 'container' is running
        self.claims = _Dict()
        self._ids = itertools.count()
        self.exception = types.SimpleNamespace(ConnectionError=ConnectionError, ServiceError=RuntimeError)
        self.Image = _Chainable()
        self.Volume = types.SimpleNamespace(from_name=lambda *a, **k: types.SimpleNamespace(
            commit=lambda: None, reload=lambda: None))
        self.Dict = types.SimpleNamespace(from_name=lambda *a, **k: self.claims)
        self.FunctionCall = types.SimpleNamespace(from_id=self._from_id)
        self.live = set()

    def current_function_call_id(self):
        return self.call_id

    def _from_id(self, call_id):
        def get(timeout=None):
            if call_id in self.live:
                raise TimeoutError
            return {}
        return types.SimpleNamespace(get=get)

    def App(self, name):
        stub = self

        class _Function:
            def __init__(self, f):
                self.f = f

            def spawn(self, *args, **kwargs):
                call_id = f"fc-{next(stub._ids)}"
                stub.spawned.append((call_id, self, args, kwargs))
                return types.SimpleNamespace(object_id=call_id)

        return types.SimpleNamespace(function=lambda **k: _Function, local_entrypoint=lambda **k: (lambda f: f))


@pytest.fixture
def chain(tmp_path, monkeypatch):
    stub = StubModal()
    monkeypatch.setitem(sys.modules, "modal", stub)
    monkeypatch.syspath_prepend(str(ROOT))
    monkeypatch.chdir(ROOT)
    sys.modules.pop("modal_app", None)
    app = importlib.import_module("modal_app")
    monkeypatch.setattr(app, "RESULTS_ROOT", tmp_path / "results")
    monkeypatch.setattr(app, "PROJECT_ROOT", tmp_path / "project")
    yield types.SimpleNamespace(app=app, modal=stub, out=tmp_path / "results" / "run")
    sys.modules.pop("modal_app", None)


def _project():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        data = DRIVER.encode()
        info = tarfile.TarInfo("drive.py")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def submit(chain, total=3, mode="ok", max_segments=10, call_id="fc-submit"):
    """A submission from outside (modal_submit.py script ... --max-segments N)."""
    chain.modal.call_id = call_id
    return chain.app.run_script_gpu.f(_project(), "drive.py", ["{output_dir}", str(total), mode], "run",
                                      60.0, None, max_segments=max_segments)


def run_spawned(chain):
    """Run the calls the segments spawned, in order, as Modal would; returns their results."""
    results = []
    while chain.modal.spawned:
        call_id, fn, args, kwargs = chain.modal.spawned.pop(0)
        chain.modal.call_id = call_id
        results.append(fn.f(*args, **kwargs))
    return results


def outcomes(chain):
    from modal_support import read_segments
    return [s["outcome"] for s in read_segments(chain.out)]


def test_chain_runs_to_the_end_without_a_client(chain):
    first = submit(chain)
    assert first["status"] == "paused" and first["chain"]["next_call_id"] == "fc-0"
    later = run_spawned(chain)
    assert [r["status"] for r in later] == ["paused", "success"]
    assert outcomes(chain) == ["paused", "paused", "finished"]
    assert (chain.out / "progress").read_text() == "3"
    assert "next_call_id" not in later[-1]["chain"]


def test_finished_run_is_not_run_again(chain):
    submit(chain, total=2)
    run_spawned(chain)
    again = submit(chain, total=2, call_id="fc-again")
    assert again["status"] == "skipped" and again["reason"] == "the run has finished"
    assert not chain.modal.spawned and (chain.out / "progress").read_text() == "2"


def test_duplicate_successor_runs_once(chain):
    submit(chain, total=2)
    call_id, fn, args, kwargs = chain.modal.spawned[0]
    chain.modal.spawned.append(("fc-duplicate", fn, args, kwargs))
    first, duplicate = run_spawned(chain)
    assert first["status"] == "success"
    assert duplicate["status"] == "skipped"
    assert (chain.out / "progress").read_text() == "2"


def test_resubmission_while_the_chain_runs_is_refused(chain):
    submit(chain, total=3)
    chain.modal.live.add("fc-0")       # the successor is pending on Modal
    again = submit(chain, total=3, call_id="fc-again")
    assert again["status"] == "skipped" and "fc-0 is still running" in again["reason"]


def test_crash_stops_the_chain(chain):
    submit(chain, total=3, mode="crash")
    (crashed,) = run_spawned(chain)
    assert crashed["status"] == "failed" and crashed["returncode"] == 1
    assert outcomes(chain) == ["paused", "failed"] and not chain.modal.spawned


def test_no_progress_stops_the_chain(chain):
    (chain.out).mkdir(parents=True)
    (chain.out / "progress").write_text("1")
    submit(chain, total=3, mode="stuck")
    (stuck,) = run_spawned(chain)
    assert stuck["chain"]["stop_reason"].startswith("no progress")
    assert not chain.modal.spawned


def test_max_segments_caps_the_chain(chain):
    submit(chain, total=100, max_segments=3)
    run_spawned(chain)
    assert outcomes(chain) == ["paused"] * 3
    assert (chain.out / "progress").read_text() == "3"


def test_without_max_segments_nothing_changes(chain):
    result = chain.app.run_script_gpu.f(_project(), "drive.py", ["{output_dir}", "3", "ok"], "run", 60.0, None)
    assert result["status"] == "paused" and result["chain"] is None
    assert not chain.modal.spawned and not (chain.out / "chain").exists()
