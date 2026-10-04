"""CPU regression tests for CLI resume and failure handling.

Only the GPU runtime is replaced; the real CLI, checkpoint files and summary
logic run unchanged. No model weights or datasets are downloaded.
"""

import importlib.util
import json
from pathlib import Path
import runpy
import sys
import types

import pytest


ENTRY = Path(__file__).resolve().parents[1] / "eval" / "longbench_cvsa.py"


@pytest.fixture
def cli(monkeypatch, tmp_path):
    state = types.SimpleNamespace(loads=0, calls=[], fail=set(), snapshots=[])
    output = tmp_path / "run.json"

    class Model:
        def __init__(self, *args, **kwargs):
            state.loads += 1
            self.config = types.SimpleNamespace(
                num_q_heads=4, num_kv_heads=2, head_dim=8, num_layers=2)

    class Runner:
        def __init__(self, *args, **kwargs):
            pass

        def evaluate_dataset(self, task, backend, n, **kwargs):
            state.calls.append((backend, task, n))
            if output.exists():
                state.snapshots.append(json.loads(output.read_text()))
            if (backend, task) in state.fail:
                raise RuntimeError("simulated evaluation failure")
            return 25.0, 0.0, n, ["answer"] * n, [25.0] * n

    imports = {
        "cvsa.runtime.model_loader": {"ModelForDecode": Model},
        "cvsa.runtime.dense": {"DenseAttentionBackend": lambda *args: "dense"},
        "eval.longbench_harness": {"LongBenchE2ERunner": Runner},
        "eval.baselines": {"make_backend": lambda *args: object()},
        "cvsa.backend": {"CVSABackend": object},
    }
    for name, members in imports.items():
        module = types.ModuleType(name)
        module.__dict__.update(members)
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("_longbench_run_state", ENTRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "make_backend", lambda method, *args: method)

    def argv(*extra):
        monkeypatch.setattr(sys, "argv", [
            str(ENTRY), "--model", "test-model", "--methods", "dense,fixed_5",
            "--tasks", "hotpotqa+qasper", "--max-samples", "5",
            "--output", str(output), *extra,
        ])

    def run(*extra):
        argv(*extra)
        return module.main()

    return types.SimpleNamespace(run=run, argv=argv, output=output, state=state)


@pytest.mark.parametrize("flag,value", [
    ("--max-samples", "50"), ("--model", "another-model"),
    ("--block-size", "32"), ("--temperature", "0.6"),
    ("--top-p", "0.9"), ("--sample-seed", "42"),
    ("--tasks", "hotpotqa"), ("--methods", "dense"),
])
def test_incompatible_resume_preserves_file_and_skips_model(cli, flag, value, capsys):
    assert cli.run() == 0
    original = cli.output.read_bytes()
    with pytest.raises(SystemExit) as exc:
        cli.run(flag, value)
    assert exc.value.code == 2
    assert cli.state.loads == 1
    assert cli.output.read_bytes() == original
    assert "Use a different --output" in capsys.readouterr().err


def test_compatible_legacy_resume_keeps_scores_and_stats(cli):
    assert cli.run() == 0
    saved = json.loads(cli.output.read_text())
    for key in ("temperature", "top_p", "sample_seed"):
        saved["config"].pop(key)
    saved.pop("cvsa_stats")
    saved["ssa_stats"] = {"fixed_5": {"hotpotqa": {"total_pages_attended": 123}}}
    cli.output.write_text(json.dumps(saved))
    calls = list(cli.state.calls)
    assert cli.run() == 0
    restored = json.loads(cli.output.read_text())
    assert cli.state.calls == calls
    for key in ("scores", "predictions", "per_sample_scores"):
        assert restored[key] == saved[key]
    assert restored["cvsa_stats"] == saved["ssa_stats"]
    assert restored["status"] == "complete"


def test_missing_config_is_not_overwritten(cli):
    cli.output.write_text('{"scores": {}}')
    original = cli.output.read_bytes()
    with pytest.raises(SystemExit) as exc:
        cli.run()
    assert exc.value.code == 2
    assert cli.state.loads == 0
    assert cli.output.read_bytes() == original


def test_failure_is_saved_and_only_failed_cell_is_retried(cli, capsys):
    cli.state.fail.add(("dense", "hotpotqa"))
    assert cli.run() == 1
    saved = json.loads(cli.output.read_text())
    assert saved["status"] == "incomplete"
    assert (saved["completed_cells"], saved["expected_cells"]) == (3, 4)
    assert saved["scores"]["dense"]["hotpotqa"] is None
    assert "simulated evaluation failure" in saved["errors"]["dense"]["hotpotqa"]
    # The first failing cell was saved before evaluating the next cell.
    assert cli.state.snapshots[0]["completed_cells"] == 0
    assert cli.state.snapshots[0]["errors"]["dense"]["hotpotqa"]
    output = capsys.readouterr()
    mean = next(line for line in output.out.splitlines() if line.startswith("MEAN"))
    delta = next(line for line in output.out.splitlines() if line.startswith("\u0394 vs dense"))
    assert "INCOMPLETE" in mean and "25.00" in mean
    assert "INCOMPLETE" in delta
    assert "3/4" in output.err

    cli.state.fail.clear()
    cli.state.calls.clear()
    assert cli.run() == 0
    complete = json.loads(cli.output.read_text())
    assert cli.state.calls == [("dense", "hotpotqa", 5)]
    assert complete["status"] == "complete"
    assert complete["errors"] == {}
    assert complete["scores"]["fixed_5"] == saved["scores"]["fixed_5"]


def test_cli_propagates_failure_exit_code(cli):
    cli.state.fail.add(("dense", "hotpotqa"))
    cli.argv("--methods", "dense")
    # run_path executes the actual __main__ guard and sys.exit(main()).
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(ENTRY), run_name="__main__")
    assert exc.value.code == 1
    assert json.loads(cli.output.read_text())["status"] == "incomplete"
