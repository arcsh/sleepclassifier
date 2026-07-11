#!/usr/bin/env python3
"""Bounded pipeline watchdog: caffeinate preflight, liveness polling, staged escalation."""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]  # repo root
ROOT = REPO_ROOT
sys.path.insert(0, str(REPO_ROOT / "src"))

from sleepstage.evaluation.results_report import write_results_md  # noqa: E402
from sleepstage.utils.logging import setup_logging  # noqa: E402

_run_unattended_spec = importlib.util.spec_from_file_location(
    "run_unattended", ROOT / "large-scale-test" / "scripts" / "run_unattended.py"
)
_run_unattended = importlib.util.module_from_spec(_run_unattended_spec)
assert _run_unattended_spec.loader is not None
_run_unattended_spec.loader.exec_module(_run_unattended)
check_class_distribution = _run_unattended.check_class_distribution
disk_ok = _run_unattended.disk_ok
run_sanity_gate = _run_unattended.run_sanity_gate

logger = logging.getLogger(__name__)

STALL_POLLS = 2
MAX_RUNTIME_SEC = 10 * 3600
RAW_DIR = ROOT / "data" / "raw"
MANIFEST_PATH = ROOT / "data" / "processed" / "manifest.csv"
STATUS_PATH = ROOT / "experiments" / "run_status.json"
RESULTS_PATH = ROOT / "RESULTS.md"
LOGS = ROOT / "logs"
INVESTIGATION_LOG = LOGS / "investigation.md"
FIXES_PATH = LOGS / "watchdog_fixes.json"
SKIP_LOG = LOGS / "skipped_records.log"

RSYNC_DEST = RAW_DIR / "sleep-cassette"
MNE_RAW = RAW_DIR / "sc"

CAFFEINATE_ENV = "SLEEPCLASSIFIER_WATCHDOG_CAFFEINATED"

# ponytail: injectable for dry-run / tests (fake clock, no 5min sleeps)
_clock: Callable[[], float] = time.monotonic
_sleep: Callable[[float], None] = time.sleep
_popen = subprocess.Popen
_pmset_run = lambda: subprocess.run(  # noqa: E731
    ["pmset", "-g", "assertions"], capture_output=True, text=True
)


def poll_sec() -> float:
    return float(os.environ.get("WATCHDOG_POLL_SEC", "300"))


def configure_paths(data_root: Path) -> None:
    """Point data/log/results paths at *data_root* (dry-run / tests). REPO_ROOT unchanged."""
    global RAW_DIR, MANIFEST_PATH, STATUS_PATH, RESULTS_PATH, LOGS
    global INVESTIGATION_LOG, FIXES_PATH, SKIP_LOG, RSYNC_DEST, MNE_RAW
    RAW_DIR = data_root / "data" / "raw"
    MANIFEST_PATH = data_root / "data" / "processed" / "manifest.csv"
    STATUS_PATH = data_root / "experiments" / "run_status.json"
    RESULTS_PATH = data_root / "RESULTS.md"
    LOGS = data_root / "logs"
    INVESTIGATION_LOG = LOGS / "investigation.md"
    FIXES_PATH = LOGS / "watchdog_fixes.json"
    SKIP_LOG = LOGS / "skipped_records.log"
    RSYNC_DEST = RAW_DIR / "sleep-cassette"
    MNE_RAW = RAW_DIR / "sc"


class DownloadMethod(str, Enum):
    MNE_HTTP = "mne_http"
    RSYNC = "rsync"
    WGET = "wget"


@dataclass
class Progress:
    files: int = 0
    bytes: int = 0
    rows: int = 0
    status_ts: str = ""

    def changed(self, other: Progress) -> bool:
        return (
            self.files != other.files
            or self.bytes != other.bytes
            or self.rows != other.rows
            or self.status_ts != other.status_ts
        )


@dataclass
class WatchdogState:
    started: float = field(default_factory=lambda: _clock())
    auto_fixes: list[str] = field(default_factory=list)
    download_method: DownloadMethod = DownloadMethod.MNE_HTTP
    download_attempts: dict[str, int] = field(default_factory=dict)
    download_errors: list[str] = field(default_factory=list)
    skipped_records: int = 0
    sanity_passed: bool = False
    sweep_restarts: int = 0


def elapsed(state: WatchdogState) -> float:
    return _clock() - state.started


def save_fixes(state: WatchdogState) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    FIXES_PATH.write_text(json.dumps(state.auto_fixes, indent=2))


def log_investigation(
    symptom: str,
    diagnosis: str,
    action: str,
    outcome: str,
) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    block = (
        f"\n## {ts}\n\n"
        f"- **Symptom:** {symptom}\n"
        f"- **Diagnosis:** {diagnosis}\n"
        f"- **Action:** {action}\n"
        f"- **Outcome:** {outcome}\n"
    )
    with INVESTIGATION_LOG.open("a") as f:
        f.write(block)
    logger.info("investigation: %s → %s (%s)", symptom, action, outcome)


def write_results(state: WatchdogState, *, diagnosis: str | None = None) -> None:
    save_fixes(state)
    write_results_md(
        RESULTS_PATH,
        ROOT / "experiments",
        auto_fixes=state.auto_fixes,
        watchdog_diagnosis=diagnosis,
    )


def ensure_caffeinate() -> None:
    if os.environ.get(CAFFEINATE_ENV):
        return
    env = os.environ.copy()
    env[CAFFEINATE_ENV] = "1"
    cmd = ["caffeinate", "-s", sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    logger.info("Re-exec under caffeinate -s")
    raise SystemExit(subprocess.run(cmd, cwd=ROOT, env=env).returncode)


def verify_caffeinate() -> tuple[bool, str]:
    proc = _pmset_run()
    out = proc.stdout + proc.stderr
    ok = "PreventUserIdleSystemSleep" in out
    return ok, out


def preflight(state: WatchdogState) -> bool:
    ensure_caffeinate()
    holding, pmset_out = verify_caffeinate()
    if not holding:
        log_investigation(
            "preflight caffeinate",
            "pmset shows no PreventUserIdleSystemSleep assertion",
            "abort before pipeline",
            f"pmset output tail:\n{pmset_out[-800:]}",
        )
        write_results(
            state,
            diagnosis=(
                "caffeinate -s did not register PreventUserIdleSystemSleep "
                "in pmset -g assertions"
            ),
        )
        return False
    log_investigation(
        "preflight caffeinate",
        "pmset shows PreventUserIdleSystemSleep",
        "caffeinate -s confirmed",
        "preflight OK",
    )

    ok, free_gb = disk_ok()
    if not ok:
        log_investigation(
            "disk space",
            f"only {free_gb:.1f} GB free",
            "abort",
            "below minimum",
        )
        write_results(state, diagnosis=f"Insufficient disk: {free_gb:.1f} GB free")
        return False
    LOGS.mkdir(exist_ok=True)
    write_results(state)
    return True


def raw_progress() -> Progress:
    if not RAW_DIR.exists():
        return Progress()
    files = [p for p in RAW_DIR.rglob("*") if p.is_file()]
    return Progress(files=len(files), bytes=sum(p.stat().st_size for p in files))


def manifest_progress() -> Progress:
    if not MANIFEST_PATH.exists():
        return Progress()
    try:
        return Progress(rows=len(pd.read_csv(MANIFEST_PATH)))
    except Exception:
        return Progress()


def sweep_progress() -> Progress:
    if not STATUS_PATH.exists():
        return Progress()
    try:
        status = json.loads(STATUS_PATH.read_text())
    except json.JSONDecodeError:
        return Progress(status_ts=str(STATUS_PATH.stat().st_mtime))
    ts = status.get("updated") or status.get("started", "")
    combos = status.get("combos", [])
    if combos:
        ts = max(ts, max(c.get("finished", "") for c in combos))
    return Progress(status_ts=ts or str(STATUS_PATH.stat().st_mtime))


def kill_proc(proc: subprocess.Popen[Any] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def build_download_cmd(method: DownloadMethod) -> list[str]:
    if method == DownloadMethod.MNE_HTTP:
        return [sys.executable, str(ROOT / "scripts" / "download_data.py"), "--subset", "sc"]
    if method == DownloadMethod.RSYNC:
        RSYNC_DEST.mkdir(parents=True, exist_ok=True)
        return [
            "rsync",
            "-Cavz",
            "physionet.org::sleep-edfx/sleep-cassette/",
            str(RSYNC_DEST) + "/",
        ]
    RSYNC_DEST.mkdir(parents=True, exist_ok=True)
    return [
        "wget",
        "-r",
        "-np",
        "-c",
        "-P",
        str(RSYNC_DEST),
        "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/",
    ]


def preprocess_raw_dir(state: WatchdogState) -> Path:
    if state.download_method in (DownloadMethod.RSYNC, DownloadMethod.WGET):
        return RSYNC_DEST
    return MNE_RAW


def port_873_blocked(stderr: str) -> bool:
    low = stderr.lower()
    return ("connection refused" in low and "873" in low) or ("rsync" in low and "failed" in low)


def monitor_subprocess(
    proc: subprocess.Popen[Any],
    progress_fn: Callable[[], Progress],
    state: WatchdogState,
    stage: str,
    *,
    log_path: Path | None = None,
) -> tuple[int, str, bool]:
    """Poll liveness every poll_sec(). Returns (rc, output_tail, stalled)."""
    prev = progress_fn()
    stall_polls = 0
    output_chunks: list[str] = []

    while proc.poll() is None:
        if elapsed(state) >= MAX_RUNTIME_SEC:
            log_investigation(
                f"{stage} hard stop",
                f"watchdog runtime >= {MAX_RUNTIME_SEC // 3600}h",
                "kill subprocess",
                "timeout",
            )
            kill_proc(proc)
            return -9, "".join(output_chunks)[-4000:], True

        _sleep(poll_sec())
        cur = progress_fn()
        if cur.changed(prev):
            stall_polls = 0
            prev = cur
            continue
        stall_polls += 1
        log_investigation(
            f"{stage} stall poll {stall_polls}/{STALL_POLLS}",
            f"no forward progress since {prev}",
            "observe",
            "waiting" if stall_polls < STALL_POLLS else "stalled",
        )
        if stall_polls >= STALL_POLLS:
            kill_proc(proc)
            if log_path and log_path.exists():
                output_chunks.append(log_path.read_text()[-2000:])
            return -1, "".join(output_chunks)[-4000:], True

    out = ""
    if log_path and log_path.exists():
        out = log_path.read_text()[-4000:]
    return proc.returncode or 0, out, False


def next_download_method(
    state: WatchdogState, stalled: bool, rc: int, err: str
) -> DownloadMethod | None:
    method = state.download_method
    if method == DownloadMethod.MNE_HTTP:
        restarts = state.download_attempts.get("mne_restart", 0)
        if stalled and restarts == 0:
            state.download_attempts["mne_restart"] = 1
            return DownloadMethod.MNE_HTTP
        if stalled or rc != 0:
            return DownloadMethod.RSYNC
        return None
    if method == DownloadMethod.RSYNC:
        if rc != 0 or port_873_blocked(err):
            return DownloadMethod.WGET
        return None
    return None


def _record_download_fix(state: WatchdogState, nxt: DownloadMethod, stalled: bool) -> str:
    if nxt == DownloadMethod.MNE_HTTP:
        fix = "download restarted MNE HTTP after stall"
    elif nxt == DownloadMethod.RSYNC:
        fix = "download escalated to rsync after 2 HTTP stalls"
    else:
        fix = "download escalated to wget after rsync failure"
    if fix not in state.auto_fixes:
        state.auto_fixes.append(fix)
    if nxt == DownloadMethod.MNE_HTTP:
        wait = 60 * state.download_attempts.get("mne_restart", 1)
        return f"restart MNE HTTP after {wait}s backoff"
    if nxt == DownloadMethod.RSYNC:
        return "switch to PhysioNet rsync mirror (port 873)"
    return "fall back to wget over HTTPS (-c resume)"


def run_download_stage(state: WatchdogState) -> bool:
    method = state.download_method
    while True:
        if elapsed(state) >= MAX_RUNTIME_SEC:
            hard_stop_before_sanity(state, "10h cap during download")
            return False

        cmd = build_download_cmd(method)
        log_path = LOGS / f"download_{method.value}.log"
        LOGS.mkdir(exist_ok=True)
        log_investigation(
            "download start",
            f"method={method.value}",
            " ".join(cmd),
            "launched",
        )

        with log_path.open("w") as logf:
            proc = _popen(cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT)

        rc, err_tail, stalled = monitor_subprocess(
            proc, raw_progress, state, f"download:{method.value}", log_path=log_path
        )
        if log_path.exists():
            err_tail = log_path.read_text()[-4000:]

        if rc == 0 and not stalled:
            state.download_method = method
            log_investigation(
                "download complete",
                f"method={method.value}",
                "none",
                f"ok files={raw_progress().files}",
            )
            if method != DownloadMethod.MNE_HTTP and not any(
                "download escalated" in f for f in state.auto_fixes
            ):
                state.auto_fixes.append(f"download escalated to {method.value}")
            return True

        state.download_errors.append(
            f"{method.value}: rc={rc} stalled={stalled}\n{err_tail[-1500:]}"
        )
        nxt = next_download_method(state, stalled, rc, err_tail)
        if nxt is None:
            diagnosis = "All download methods exhausted.\n\n" + "\n\n---\n\n".join(
                state.download_errors
            )
            log_investigation(
                "download failed",
                diagnosis,
                "stop pipeline",
                "FAILED",
            )
            write_results(state, diagnosis=diagnosis)
            return False

        action = _record_download_fix(state, nxt, stalled)
        log_investigation(
            f"download {'stalled' if stalled else 'failed'} ({method.value})",
            err_tail[-500:] or f"exit {rc}",
            action,
            "escalating",
        )
        method = nxt
        state.download_method = method
        if nxt == DownloadMethod.MNE_HTTP:
            _sleep(60 * state.download_attempts.get("mne_restart", 1))


def run_preprocess_stage(state: WatchdogState) -> bool:
    raw_dir = preprocess_raw_dir(state)
    skip_log = SKIP_LOG
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "preprocess.py"),
        "--raw-dir",
        str(raw_dir),
        "--skip-log",
        str(skip_log),
    ]
    log_path = LOGS / "preprocess.log"

    while True:
        if elapsed(state) >= MAX_RUNTIME_SEC:
            hard_stop_before_sanity(state, "10h cap during preprocess")
            return False

        with log_path.open("w") as logf:
            proc = _popen(cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT)
        rc, err_tail, stalled = monitor_subprocess(
            proc, manifest_progress, state, "preprocess", log_path=log_path
        )
        if log_path.exists():
            err_tail = log_path.read_text()[-4000:]

        if rc == 0:
            if skip_log.exists():
                state.skipped_records = skip_log.read_text().count("id=")
                if state.skipped_records:
                    fix = f"{state.skipped_records} records skipped, " "see skipped_records.log"
                    if fix not in state.auto_fixes:
                        state.auto_fixes.append(fix)
            log_investigation("preprocess complete", "manifest written", "none", "ok")
            return True

        if rc == 2:
            log_investigation(
                "preprocess systemic skip",
                err_tail[-800:],
                "stop — same error class >5 records",
                "FAILED",
            )
            write_results(
                state,
                diagnosis=(f"Systemic preprocess failure (>{5} same error class).\n\n{err_tail}"),
            )
            return False

        if stalled:
            log_investigation(
                "preprocess stalled",
                "manifest row count flat for 10 min",
                "restart preprocess (skips existing npz)",
                "retry",
            )
            fix = "preprocess restarted after stall"
            if fix not in state.auto_fixes:
                state.auto_fixes.append(fix)
            continue

        log_investigation(
            "preprocess crash",
            err_tail[-800:],
            "stop",
            f"exit {rc}",
        )
        write_results(state, diagnosis=f"Preprocess failed rc={rc}\n\n{err_tail}")
        return False


def hard_stop_before_sanity(state: WatchdogState, reason: str) -> None:
    log_investigation("hard stop", reason, "abort before sweep", "FAILED")
    write_results(state, diagnosis=reason)


def run_sanity_and_sweep(state: WatchdogState) -> int:
    manifest = MANIFEST_PATH
    dist_ok, dist_report = check_class_distribution(manifest)
    (LOGS / "preprocess_check.log").write_text(
        f"{datetime.now(timezone.utc).isoformat()}\n{dist_report}\n"
    )
    if not dist_ok:
        hard_stop_before_sanity(state, f"preprocess check failed:\n{dist_report}")
        return 1

    gate_ok, gate_report = run_sanity_gate()
    (LOGS / "sanity_gate.log").write_text(
        f"{datetime.now(timezone.utc).isoformat()}\n{gate_report}\n"
    )
    if not gate_ok:
        hard_stop_before_sanity(state, f"sanity gate failed:\n{gate_report}")
        return 1

    state.sanity_passed = True
    log_investigation("sanity gate", gate_report, "pass", "launch sweep")

    sweep_log = LOGS / "full_run.log"
    while True:
        if elapsed(state) >= MAX_RUNTIME_SEC:
            hard_stop_before_sanity(
                state,
                f"10h cap hit after sanity gate; sweep incomplete at "
                f"{elapsed(state)/3600:.1f}h",
            )
            return 1

        with sweep_log.open("a" if sweep_log.exists() else "w") as logf:
            if state.sweep_restarts:
                logf.write(
                    f"\n--- watchdog sweep restart {state.sweep_restarts} "
                    f"{datetime.now(timezone.utc).isoformat()} ---\n"
                )
            proc = _popen(
                [sys.executable, str(ROOT / "large-scale-test" / "scripts" / "run_full_sweep.py")],
                cwd=ROOT,
                stdout=logf,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"},
            )
        (LOGS / "sweep_pid.txt").write_text(str(proc.pid))
        rc, _, stalled = monitor_subprocess(
            proc, sweep_progress, state, "sweep", log_path=sweep_log
        )

        if rc == 0 and not stalled:
            write_results(state)
            log_investigation("sweep", "run_full_sweep exited 0", "done", "SUCCESS/PARTIAL")
            return 0

        if stalled and state.sweep_restarts < 1:
            state.sweep_restarts += 1
            fix = "sweep restarted after 10 min stall"
            if fix not in state.auto_fixes:
                state.auto_fixes.append(fix)
            log_investigation(
                "sweep stalled",
                "run_status.json timestamp flat",
                "kill + restart run_full_sweep (resume-safe)",
                "retry once",
            )
            continue

        write_results(state)
        log_investigation("sweep", f"rc={rc} stalled={stalled}", "stop", "FAILED/PARTIAL")
        return rc if rc != 0 else 1


# --- dry-run / self-check (mocked failures, injectable clock) ---


class _StallProc:
    """Subprocess that never makes progress until killed by monitor_subprocess."""

    def __init__(self, cmd: list[str], log_rc: int = -1):
        self.args = cmd
        self.cmd = cmd
        self.returncode: int | None = None
        self._log_rc = log_rc
        self.pid = os.getpid()

    def poll(self) -> int | None:
        return self.returncode

    def send_signal(self, sig: int) -> None:
        self.returncode = self._log_rc

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            self.returncode = self._log_rc
        return self.returncode

    def kill(self) -> None:
        self.returncode = -9


class _ExitProc:
    """Subprocess that exits immediately with rc."""

    def __init__(self, cmd: list[str], rc: int, log_text: str = ""):
        self.args = cmd
        self.cmd = cmd
        self.returncode = rc
        self.pid = os.getpid()
        self._log_text = log_text

    def poll(self) -> int:
        return self.returncode

    def send_signal(self, sig: int) -> None:
        return None

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode

    def kill(self) -> None:
        return None


def _reset_hooks() -> None:
    global _clock, _sleep, _popen, _pmset_run
    _clock = time.monotonic
    _sleep = time.sleep
    _popen = subprocess.Popen
    _pmset_run = lambda: subprocess.run(  # noqa: E731
        ["pmset", "-g", "assertions"], capture_output=True, text=True
    )


def run_dry_run() -> None:
    """Exercise escalation paths with mocks (no network, no real 10h wait)."""
    import tempfile

    orig_paths = {
        "RAW_DIR": RAW_DIR,
        "MANIFEST_PATH": MANIFEST_PATH,
        "STATUS_PATH": STATUS_PATH,
        "RESULTS_PATH": RESULTS_PATH,
        "LOGS": LOGS,
        "INVESTIGATION_LOG": INVESTIGATION_LOG,
        "FIXES_PATH": FIXES_PATH,
        "SKIP_LOG": SKIP_LOG,
        "RSYNC_DEST": RSYNC_DEST,
        "MNE_RAW": MNE_RAW,
    }

    def restore_paths() -> None:
        global RAW_DIR, MANIFEST_PATH, STATUS_PATH, RESULTS_PATH, LOGS
        global INVESTIGATION_LOG, FIXES_PATH, SKIP_LOG, RSYNC_DEST, MNE_RAW
        RAW_DIR = orig_paths["RAW_DIR"]
        MANIFEST_PATH = orig_paths["MANIFEST_PATH"]
        STATUS_PATH = orig_paths["STATUS_PATH"]
        RESULTS_PATH = orig_paths["RESULTS_PATH"]
        LOGS = orig_paths["LOGS"]
        INVESTIGATION_LOG = orig_paths["INVESTIGATION_LOG"]
        FIXES_PATH = orig_paths["FIXES_PATH"]
        SKIP_LOG = orig_paths["SKIP_LOG"]
        RSYNC_DEST = orig_paths["RSYNC_DEST"]
        MNE_RAW = orig_paths["MNE_RAW"]

    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            configure_paths(root)
            os.environ["WATCHDOG_POLL_SEC"] = "0.001"
            (root / "experiments").mkdir(parents=True, exist_ok=True)

            fake_t = [0.0]

            def advance_clock(secs: float) -> None:
                fake_t[0] += secs

            _reset_hooks()
            global _clock, _sleep, _popen, _pmset_run
            _clock = lambda: fake_t[0]
            _sleep = lambda s: advance_clock(s)

            pmset_good = (
                "Assertion status system-wide:\n"
                "   PreventUserIdleSystemSleep    1\n"
                "   pid 123(caffeinate): PreventUserIdleSystemSleep\n"
            )

            def scenario_download_ladder() -> None:
                global _popen
                fake_t[0] = 0.0
                RAW_DIR.mkdir(parents=True, exist_ok=True)
                LOGS.mkdir(parents=True, exist_ok=True)
                INVESTIGATION_LOG.write_text("# dry-run download\n")
                scripts: list[Any] = [
                    lambda c: _StallProc(c),
                    lambda c: _StallProc(c),
                    lambda c: _ExitProc(c, 1, "connection refused on port 873"),
                    lambda c: _ExitProc(c, 1, "wget failed"),
                ]
                launched: list[str] = []

                def popen_factory(cmd, **kwargs):
                    launched.append(cmd[0])
                    return scripts.pop(0)(cmd)

                _popen = popen_factory
                state = WatchdogState(started=fake_t[0])
                ok = run_download_stage(state)
                assert not ok
                assert launched.count(sys.executable) == 2
                assert "rsync" in launched
                assert "wget" in launched
                body = RESULTS_PATH.read_text()
                assert "## Watchdog diagnosis" in body
                assert "All download methods exhausted" in body
                assert "mne_http" in body
                inv = INVESTIGATION_LOG.read_text()
                assert "download escalated to rsync after 2 HTTP stalls" in " ".join(
                    state.auto_fixes
                )
                assert "Symptom:" in inv

            def scenario_download_rsync_success() -> None:
                global _popen
                fake_t[0] = 0.0
                LOGS.mkdir(parents=True, exist_ok=True)
                INVESTIGATION_LOG.write_text("# dry-run rsync ok\n")
                scripts = [
                    lambda c: _StallProc(c),
                    lambda c: _StallProc(c),
                    lambda c: _ExitProc(c, 0),
                ]

                def popen_factory(cmd, **kwargs):
                    proc = scripts.pop(0)(cmd)
                    if cmd[0] == "rsync":
                        RSYNC_DEST.mkdir(parents=True, exist_ok=True)
                        (RSYNC_DEST / "dummy.edf").write_bytes(b"x" * 10)
                    return proc

                _popen = popen_factory
                state = WatchdogState(started=fake_t[0])
                assert run_download_stage(state)
                assert state.download_method == DownloadMethod.RSYNC

            def scenario_caffeinate_fail() -> None:
                global _pmset_run
                _pmset_run = lambda: type(  # noqa: E731
                    "R", (), {"stdout": "No sleep assertions", "stderr": ""}
                )()
                os.environ[CAFFEINATE_ENV] = "1"
                try:
                    state = WatchdogState(started=fake_t[0])
                    LOGS.mkdir(parents=True, exist_ok=True)
                    assert not preflight(state)
                    assert "PreventUserIdleSystemSleep" in RESULTS_PATH.read_text()
                finally:
                    os.environ.pop(CAFFEINATE_ENV, None)

            def scenario_caffeinate_ok() -> None:
                global _pmset_run
                _pmset_run = lambda: type(  # noqa: E731
                    "R", (), {"stdout": pmset_good, "stderr": ""}
                )()
                os.environ[CAFFEINATE_ENV] = "1"
                try:
                    state = WatchdogState(started=fake_t[0])
                    LOGS.mkdir(parents=True, exist_ok=True)
                    assert preflight(state)
                    assert "PreventUserIdleSystemSleep" in INVESTIGATION_LOG.read_text()
                finally:
                    os.environ.pop(CAFFEINATE_ENV, None)

            def scenario_hard_stop() -> None:
                fake_t[0] = MAX_RUNTIME_SEC + 1
                state = WatchdogState(started=0.0)
                LOGS.mkdir(parents=True, exist_ok=True)
                INVESTIGATION_LOG.write_text("# hard stop\n")
                assert not run_download_stage(state)
                assert "10h cap" in INVESTIGATION_LOG.read_text()

            def scenario_preprocess_skip() -> None:
                pp_spec = importlib.util.spec_from_file_location(
                    "preprocess", REPO_ROOT / "scripts" / "preprocess.py"
                )
                pp = importlib.util.module_from_spec(pp_spec)
                assert pp_spec.loader is not None
                pp_spec.loader.exec_module(pp)
                skip_log = LOGS / "skipped_records.log"
                skip_log.parent.mkdir(parents=True, exist_ok=True)
                exc = FileNotFoundError("missing hypnogram")
                err_class = pp.classify_preprocess_error(exc)
                assert pp.is_skippable_record_error(exc)
                with skip_log.open("a") as f:
                    f.write(f"ts id=SC4002E0 class={err_class} error={exc}\n")
                assert skip_log.exists()
                assert "id=SC4002E0" in skip_log.read_text()

            def scenario_preprocess_systemic() -> None:
                global _popen
                pp_spec = importlib.util.spec_from_file_location(
                    "preprocess", REPO_ROOT / "scripts" / "preprocess.py"
                )
                pp = importlib.util.module_from_spec(pp_spec)
                assert pp_spec.loader is not None
                pp_spec.loader.exec_module(pp)
                assert pp.SYSTEMIC_SKIP_LIMIT == 5
                assert pp.SKIP_EXIT_CODE == 2
                LOGS.mkdir(parents=True, exist_ok=True)
                INVESTIGATION_LOG.write_text("# systemic\n")
                (LOGS / "preprocess.log").write_text("systemic bad_edf_header x6\n")

                def popen_factory(cmd, **kwargs):
                    return _ExitProc(cmd, pp.SKIP_EXIT_CODE, "systemic bad_edf_header")

                _popen = popen_factory
                state = WatchdogState(started=fake_t[0])
                assert not run_preprocess_stage(state)
                body = RESULTS_PATH.read_text()
                assert "Systemic preprocess failure" in body
                assert "systemic" in INVESTIGATION_LOG.read_text().lower()

            def scenario_sweep_stall() -> None:
                global _popen
                fake_t[0] = 0.0
                LOGS.mkdir(parents=True, exist_ok=True)
                INVESTIGATION_LOG.write_text("# sweep\n")
                STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
                STATUS_PATH.write_text(json.dumps({"started": "t0", "updated": "t0", "combos": []}))

                def popen_factory(cmd, **kwargs):
                    return _StallProc(cmd)

                _popen = popen_factory
                state = WatchdogState(started=fake_t[0])
                sweep_log = LOGS / "full_run.log"
                proc = _popen([sys.executable, "sweep"], stdout=subprocess.PIPE)
                rc, _, stalled = monitor_subprocess(
                    proc, sweep_progress, state, "sweep", log_path=sweep_log
                )
                assert stalled
                fix = "sweep restarted after 10 min stall"
                state.sweep_restarts = 1
                state.auto_fixes.append(fix)
                log_investigation("sweep stalled", "flat", "restart", "retry once")
                assert fix in state.auto_fixes
                assert "Symptom:" in INVESTIGATION_LOG.read_text()

            def scenario_hard_stop_before_sanity() -> None:
                fake_t[0] = MAX_RUNTIME_SEC + 1
                state = WatchdogState(started=0.0)
                LOGS.mkdir(parents=True, exist_ok=True)
                hard_stop_before_sanity(state, "10h cap before sanity gate — not launching sweep")
                assert "not launching sweep" in RESULTS_PATH.read_text()

            scenario_download_ladder()
            scenario_download_rsync_success()
            scenario_caffeinate_fail()
            scenario_caffeinate_ok()
            scenario_hard_stop()
            scenario_hard_stop_before_sanity()
            scenario_preprocess_skip()
            scenario_preprocess_systemic()
            scenario_sweep_stall()
    finally:
        restore_paths()
        _reset_hooks()
        os.environ.pop("WATCHDOG_POLL_SEC", None)

    print("watchdog dry-run OK")


def _self_check() -> None:
    """ponytail: assert helpers without network."""
    a, b = Progress(files=1, bytes=10), Progress(files=2, bytes=10)
    assert a.changed(b)
    assert not b.changed(b)
    assert port_873_blocked("rsync: connection refused on port 873")
    assert poll_sec() == 300.0 or os.environ.get("WATCHDOG_POLL_SEC")

    st = WatchdogState()
    st.download_method = DownloadMethod.MNE_HTTP
    st.download_attempts["mne_restart"] = 0
    assert next_download_method(st, True, -1, "") == DownloadMethod.MNE_HTTP
    st.download_attempts["mne_restart"] = 1
    assert next_download_method(st, True, -1, "") == DownloadMethod.RSYNC
    st.download_method = DownloadMethod.RSYNC
    assert next_download_method(st, False, 1, "connection refused 873") == DownloadMethod.WGET

    ok, _ = verify_caffeinate() if sys.platform == "darwin" else (True, "")
    if sys.platform == "darwin" and os.environ.get(CAFFEINATE_ENV):
        assert ok, "pmset must show PreventUserIdleSystemSleep under caffeinate -s"

    run_dry_run()
    print("watchdog self-check OK")


def main() -> None:
    if "--self-check" in sys.argv:
        _self_check()
        return
    if "--dry-run" in sys.argv:
        run_dry_run()
        return

    setup_logging()
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    state = WatchdogState()
    INVESTIGATION_LOG.write_text(
        f"# Watchdog investigation log\n\nStarted {datetime.now(timezone.utc).isoformat()}\n"
    )

    if not preflight(state):
        raise SystemExit(1)

    if elapsed(state) >= MAX_RUNTIME_SEC:
        hard_stop_before_sanity(state, "10h cap before download")
        raise SystemExit(1)

    if not run_download_stage(state):
        raise SystemExit(1)

    if elapsed(state) >= MAX_RUNTIME_SEC:
        hard_stop_before_sanity(state, "10h cap before preprocess")
        raise SystemExit(1)

    if not run_preprocess_stage(state):
        raise SystemExit(1)

    if elapsed(state) >= MAX_RUNTIME_SEC:
        hard_stop_before_sanity(state, "10h cap before sanity gate — not launching sweep")
        raise SystemExit(1)

    raise SystemExit(run_sanity_and_sweep(state))


if __name__ == "__main__":
    main()
