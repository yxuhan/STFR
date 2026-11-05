"""Checked subprocess stages, live tee logs, atomic status, and conservative resume."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import codecs
import json
import os
from pathlib import Path
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass
class Step:
    name: str
    command: list
    cwd: Path
    inputs: list = field(default_factory=list)
    outputs: list = field(default_factory=list)
    fresh: bool = True
    reset: list = field(default_factory=list)
    mkdir: list = field(default_factory=list)
    code_inputs: list = field(default_factory=list)
    mutable_inputs: list = field(default_factory=list)


class StageError(RuntimeError):
    pass


class Plan:
    """Stage-neutral command construction; contains no algorithm/stage logic."""

    def __init__(self, project):
        self.project = Path(project)
        self.steps = []

    def python(
        self, name, script, args, inputs=(), outputs=(), reset=(), fresh=True, mkdir=(), cwd=None
    ):
        script = self.project / script
        self.steps.append(
            Step(
                name,
                [sys.executable, "-u", str(script), *map(str, args)],
                cwd or script.parent,
                list(inputs),
                list(outputs),
                fresh,
                list(reset),
                list(mkdir),
            )
        )

    def command(self, name, command, inputs, outputs, reset=(), fresh=True, mkdir=(), cwd=None):
        self.steps.append(
            Step(
                name,
                list(map(str, command)),
                cwd or self.project,
                list(inputs),
                list(outputs),
                fresh,
                list(reset),
                list(mkdir),
            )
        )


def stage_main(parser, builder, stage_name, project):
    """Public plan-JSON protocol or independent execution of one stage."""
    from dataclasses import asdict

    parser.add_argument(
        "--plan_json", action="store_true", help="Print this module execution plan as JSON"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    options = parser.parse_args()
    options.save_root = str(Path(options.data_root).resolve())
    if not options.device.isdigit():
        parser.error("--device must select one physical CUDA device")
    steps = builder(options)
    if options.plan_json:
        print(
            json.dumps(dict(stage=stage_name, steps=[asdict(step) for step in steps]), default=str)
        )
        return 0
    workspace = Path(options.save_root).parent if stage_name == "tex" else Path(options.save_root)
    try:
        Runner(
            workspace, project, options.resume, env={"CUDA_VISIBLE_DEVICES": options.device}
        ).run(steps, vars(options))
        return 0
    except KeyboardInterrupt:
        return 130
    except (StageError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1


def fingerprints(paths):
    result = {}
    for item in paths:
        path = Path(item)
        if path.is_file():
            entries = [path]
        elif path.is_dir():
            entries = sorted(p for p in path.rglob("*") if p.is_file())
        else:
            raise FileNotFoundError(f"Missing artifact: {path}")
        if (
            not entries
            or not any(p.stat().st_size for p in entries)
            or (path.is_file() and path.stat().st_size == 0)
        ):
            raise ValueError(f"Empty artifact: {path}")
        # Directories may legitimately contain empty auxiliary files. Only the
        # declared file outputs themselves must be nonempty.
        result[str(path.resolve())] = [
            (str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in entries
        ]
    return result


def code_fingerprint(project):
    digest = hashlib.sha256()
    for directory in ["", "refinement", "registration", "reconstruction", "texture"]:
        base = Path(project) / directory
        # Own stage logic, excluding external model/vendor trees.
        for path in sorted(base.glob("*.py")):
            digest.update(str(path.relative_to(project)).encode())
            digest.update(path.read_bytes())
    for path in sorted((Path(project) / "registration/align").rglob("*.py")):
        digest.update(str(path.relative_to(project)).encode())
        digest.update(path.read_bytes())
    for path in sorted((Path(project) / "refinement/select_frame").glob("*.py")):
        digest.update(path.read_bytes())
    calibration = Path(project) / "misc/metahuman_transform.json"
    if calibration.exists():
        digest.update(calibration.read_bytes())
    return digest.hexdigest()


class Runner:
    def __init__(self, workspace, project, resume=False, env=None, heartbeat=30):
        self.workspace = Path(workspace).resolve()
        self.project = Path(project).resolve()
        self.resume = resume
        self.heartbeat = heartbeat
        self.env = os.environ.copy()
        self.env.update(env or {})
        self.env["PYTHONUNBUFFERED"] = "1"
        self.env["PYTHONPATH"] = str(self.project) + os.pathsep + self.env.get("PYTHONPATH", "")
        self.code_hash = code_fingerprint(self.project)

    def run(self, steps, options=None):
        self.workspace.mkdir(parents=True, exist_ok=True)
        lock = self.workspace / ".pipeline.lock"
        try:
            if lock.exists():
                owner = json.loads(lock.read_text())
                if owner["host"] == socket.gethostname():
                    try:
                        os.kill(owner["pid"], 0)
                    except ProcessLookupError:
                        lock.unlink()
            descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            raise StageError(f"Another run owns this workspace; inspect {lock}")
        with os.fdopen(descriptor, "w") as stream:
            json.dump(dict(pid=os.getpid(), host=socket.gethostname()), stream)
        try:
            return self._run(steps, options)
        finally:
            lock.unlink(missing_ok=True)

    def _run(self, steps, options):
        latest_status = self.workspace / "pipeline_status.json"
        previous = (
            json.loads(latest_status.read_text()) if self.resume and latest_status.exists() else {}
        )
        old_steps = {step["name"]: step for step in previous.get("steps", [])}
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:6]
        self.log_dir = self.workspace / "logs" / run_id
        self.log_dir.mkdir(parents=True)
        self.state = dict(
            run_id=run_id,
            status="running",
            started=now(),
            workspace=str(self.workspace),
            log_dir=str(self.log_dir),
            options=options or {},
            code_hash=self.code_hash,
            current_step=None,
            steps=[],
        )
        for number, step in enumerate(steps, 1):
            self.state["steps"].append(
                dict(
                    name=step.name,
                    number=number,
                    status="pending",
                    command=list(map(str, step.command)),
                    cwd=str(step.cwd),
                    inputs=list(map(str, step.inputs)),
                    outputs=list(map(str, step.outputs)),
                    log=str(self.log_dir / f"{number:02d}_{step.name}.log"),
                )
            )
        atomic_json(
            self.workspace / "logs/latest.json", dict(run_id=run_id, directory=str(self.log_dir))
        )
        self.master = (self.log_dir / "pipeline.log").open("w", encoding="utf-8")
        self.events = (self.log_dir / "events.jsonl").open("w", encoding="utf-8")
        reuse = self.resume
        try:
            self._message(f"RUN {run_id} | {len(steps)} steps | logs: {self.log_dir}")
            for step, entry in zip(steps, self.state["steps"]):
                self.state["current_step"] = step.name
                started = time.monotonic()
                entry.update(status="running", started=now())
                self._event("step_start", entry)
                self._message(f"[{entry['number']:02d}/{len(steps):02d}] RUN {step.name}")
                try:
                    inputs = fingerprints(step.inputs)
                    mutable = {
                        str(Path(p).resolve()) for p in [*step.outputs, *step.mutable_inputs]
                    }
                    signature_inputs = {
                        key: value for key, value in inputs.items() if key not in mutable
                    }
                    signature = dict(
                        command=entry["command"],
                        cwd=entry["cwd"],
                        inputs=signature_inputs,
                        cuda_visible_devices=self.env.get("CUDA_VISIBLE_DEVICES", ""),
                        implementation=self._step_hash(step),
                    )
                    # JSON roundtrip makes tuples compare identically after loading.
                    signature = json.loads(json.dumps(signature))
                    entry["signature"] = signature
                    old = old_steps.get(step.name, {})
                    old_signature = dict(old.get("signature", {}))
                    if "inputs" in old_signature:
                        old_signature["inputs"] = {
                            key: value
                            for key, value in old_signature["inputs"].items()
                            if key not in mutable
                        }
                    try:
                        outputs_before = fingerprints(step.outputs)
                    except (FileNotFoundError, ValueError):
                        outputs_before = None
                    if (
                        reuse
                        and old.get("status") in ("succeeded", "skipped")
                        and old_signature == signature
                        and outputs_before is not None
                        and old.get("output_fingerprints") == json.loads(json.dumps(outputs_before))
                    ):
                        entry.update(
                            status="skipped",
                            output_fingerprints=old["output_fingerprints"],
                            reused_from=previous["run_id"],
                        )
                        Path(entry["log"]).write_text(
                            f"Reused validated outputs from {previous['run_id']}\n"
                        )
                        self._message(f"  SKIP {step.name} (validated prior success)")
                    else:
                        reuse = False  # Every later selected step must be regenerated.
                        self._archive(step.reset)
                        for directory in step.mkdir:
                            Path(directory).mkdir(parents=True, exist_ok=True)
                        self._execute(step, entry)
                        outputs = fingerprints(step.outputs)
                        if step.fresh and step.outputs and outputs == outputs_before:
                            raise StageError(
                                "Command returned success but declared outputs were not updated"
                            )
                        entry.update(status="succeeded", output_fingerprints=outputs)
                        self._message(f"  OK {step.name} ({time.monotonic() - started:.1f}s)")
                except BaseException as error:
                    if not Path(entry["log"]).exists():
                        Path(entry["log"]).write_text(str(error) + "\n", encoding="utf-8")
                    entry.update(
                        status="cancelled" if isinstance(error, KeyboardInterrupt) else "failed",
                        error=str(error),
                        elapsed_seconds=time.monotonic() - started,
                        finished=now(),
                    )
                    self.state.update(status=entry["status"], error=str(error), finished=now())
                    self._event("step_" + entry["status"], entry)
                    self._message(
                        f"FAILED {step.name}: {error}\n  step log: {entry['log']}\n  status: {latest_status}"
                    )
                    if isinstance(error, KeyboardInterrupt):
                        raise
                    raise StageError(f"{step.name}: {error}; log: {entry['log']}") from error
                entry.update(finished=now(), elapsed_seconds=time.monotonic() - started)
                self._event("step_" + entry["status"], entry)
            self.state.update(status="succeeded", current_step=None, finished=now())
            self._event("run_succeeded", {})
            self._message(f"COMPLETE | status: {latest_status}")
            return self.state
        finally:
            # COLMAP updates its database/model in place in later substeps.
            # Record the final fingerprint of shared artifacts for resume.
            for step, entry in zip(steps, self.state["steps"]):
                if entry["status"] in ("succeeded", "skipped"):
                    try:
                        entry["output_fingerprints"] = fingerprints(step.outputs)
                    except (FileNotFoundError, ValueError):
                        entry["output_fingerprints"] = None
            self._save()
            self.master.close()
            self.events.close()

    def _save(self):
        self.state["updated"] = now()
        atomic_json(self.log_dir / "status.json", self.state)
        atomic_json(self.workspace / "pipeline_status.json", self.state)

    def _event(self, event, detail):
        self.events.write(
            json.dumps(
                dict(time=now(), event=event, step=self.state["current_step"], detail=detail)
            )
            + "\n"
        )
        self.events.flush()
        self._save()

    def _message(self, message):
        line = f"[{now()}] {message}\n"
        print(line, end="", flush=True)
        self.master.write(line)
        self.master.flush()

    def _archive(self, paths):
        for item in paths:
            path = Path(item).absolute()
            relative = path.relative_to(self.workspace)
            if path.exists() or path.is_symlink():
                target = self.workspace / "history" / self.state["run_id"] / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                path.rename(target)
                self._message(f"  previous artifact -> {target}")

    def _step_hash(self, step):
        digest = hashlib.sha256()
        paths = list(step.code_inputs) + [
            Path(arg) for arg in step.command if str(arg).endswith(".py") and Path(arg).is_file()
        ]
        for path in sorted(set(map(Path, paths))):
            digest.update(str(path).encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def _execute(self, step, entry):
        self._message("  command: " + repr(entry["command"]) + "\n  cwd: " + str(step.cwd))
        process = subprocess.Popen(
            entry["command"],
            cwd=step.cwd,
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=(os.name != "nt"),
        )
        entry["pid"] = process.pid
        self._save()
        chunks = queue.Queue()

        def reader():
            try:
                while True:
                    chunk = os.read(process.stdout.fileno(), 8192)
                    if not chunk:
                        break
                    chunks.put(chunk)
            finally:
                chunks.put(None)

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        last_heartbeat = time.monotonic()
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        try:
            with Path(entry["log"]).open("wb") as log:
                while True:
                    try:
                        chunk = chunks.get(timeout=min(1, self.heartbeat))
                    except queue.Empty:
                        chunk = b""
                    if chunk is None:
                        remainder = decoder.decode(b"", final=True)
                        if remainder:
                            sys.stdout.write(remainder)
                            self.master.write(remainder)
                        break
                    if chunk:
                        log.write(chunk)
                        log.flush()
                        text = decoder.decode(chunk)
                        sys.stdout.write(text)
                        sys.stdout.flush()
                        self.master.write(text)
                        self.master.flush()
                    if time.monotonic() - last_heartbeat >= self.heartbeat:
                        self._message(f"  RUNNING {step.name} | pid={process.pid}")
                        self._event("heartbeat", dict(pid=process.pid))
                        last_heartbeat = time.monotonic()
            entry["returncode"] = process.wait()
            if process.returncode:
                raise StageError(f"Command exited with code {process.returncode}")
        except BaseException:
            if process.poll() is None:
                if os.name == "nt":
                    process.terminate()
                else:
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    if os.name == "nt":
                        process.kill()
                    else:
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise
        finally:
            thread.join(timeout=2)
            process.stdout.close()
