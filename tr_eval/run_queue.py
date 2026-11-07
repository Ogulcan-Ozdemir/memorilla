"""A resumable queue of evaluation runs for the Turkish evaluation, made for Colab sessions that can disconnect.

Three subcommands, run from the repository root:

- ``plan`` prints the job table (TSV: name, script, output directory, arguments) for the translated configs that
  ``tr_eval.build_data`` writes. Edit it freely; ``run`` only reads it.
- ``run`` works through a job table in order. A job runs ``evaluate.py`` or ``evaluate_baselines.py`` with the
  Python interpreter of the queue itself, logs to ``<log_dir>/<name>.log``, and is marked done when it exits cleanly.
  Done jobs are skipped on the next ``run`` unless their command changed, so rerunning after a disconnect continues
  where the queue stopped. With ``--sync_to`` (e.g. a Google Drive folder) the results, logs and data repository
  are copied there after every job.
- ``status`` prints the state of every job: done, failed, running or pending.

Start ``run`` in the background (``nohup python -m tr_eval.run_queue run runs.tsv ... &``) and poll ``status``.
"""

import argparse
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from fnmatch import fnmatch
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

from filelock import FileLock, Timeout

from memorilla.benchmarks import BENCHMARKS, CHOICE, PERSONA_PROMPT
from memorilla.paths import HOME, resolve
from tr_eval.build_data import DEFAULT_OUTPUT_REPO, VARIANTS
from tr_eval.common import stable_key
from tr_eval.prompts import PERSONA_PROMPT_REPLY_TR, PERSONA_PROMPT_TR

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = {"evaluate": "evaluate.py", "baselines": "evaluate_baselines.py"}
TIERS = ("core", "extended", "full")
DEFAULT_TIERS = ("core",)
DEFAULT_SOURCES = ("pv4", "pmv2")
DEFAULT_CHECKPOINTS = "checkpoints"
DEFAULT_RESULTS = "results/tr"
DEFAULT_LOG_DIR = "logs/tr"
GENERAL_CHECKPOINT = "stage3"
EVAL_MAX_LENGTH = 8192
MAX_NEW_TOKENS = 384
RAG_TOP_K = 5
FULL_CONTEXT_MAX_NUM_SEQS = 2
PROMPTS = {"en_prompt": PERSONA_PROMPT, "tr_prompt": PERSONA_PROMPT_TR, "en_prompt_reply_tr": PERSONA_PROMPT_REPLY_TR}
STATE_DIR = ".queue"
LOCK_FILE = "queue.lock"
DONE, FAILED, RUNNING, PENDING = "done", "failed", "running", "pending"
COLUMNS = ("name", "script", "output_dir", "args")


@dataclass(frozen=True)
class Job:
    """One evaluation run.

    Attributes:
        name: Unique name, e.g. ``pv4/tr_prompt/tr_pv4``; also names its log and state files.
        script: Key of ``SCRIPTS``.
        output_dir: ``--output_dir`` of the script (the script adds its own subfolders below it).
        args: Further arguments of the script.
    """

    name: str
    script: str
    output_dir: str
    args: tuple[str, ...]

    def command(self, python: str) -> list[str]:
        """Build the command line.

        Args:
            python: Python interpreter.

        Returns:
            The command.
        """
        return [python, SCRIPTS[self.script], *self.args, "--output_dir", self.output_dir]

    def fingerprint(self) -> str:
        """Identify the job's command, so that a changed command reruns a done job.

        Returns:
            A hash of the script, output directory and arguments.
        """
        return stable_key(self.script, self.output_dir, list(self.args))


def evaluate_job(
    checkpoint_dir: str, checkpoint: str, prompt: str, config: str, data_repo: str, results: str, extra: list[str]
) -> Job:
    """Build a Memorilla evaluation job.

    Args:
        checkpoint_dir: Directory holding the downloaded checkpoint folders.
        checkpoint: Checkpoint folder, e.g. ``pv4``.
        prompt: Key of ``PROMPTS``.
        config: Dataset config.
        data_repo: Local data repository.
        results: Results root.
        extra: Further ``evaluate.py`` arguments.

    Returns:
        The job; results land in ``<results>/<checkpoint>/<prompt>/<config>/``.
    """
    args = [
        "--checkpoint",
        f"{checkpoint_dir}/{checkpoint}",
        "--config",
        config,
        "--scoring",
        CHOICE,
        "--data_repo",
        data_repo,
        "--system_prompt",
        PROMPTS[prompt],
        "--max_length",
        str(EVAL_MAX_LENGTH),
        "--max_new_tokens",
        str(MAX_NEW_TOKENS),
        *extra,
    ]
    return Job(f"{checkpoint}/{prompt}/{config}", "evaluate", f"{results}/{checkpoint}/{prompt}", tuple(args))


def baseline_job(method: str, source: str, config: str, data_repo: str, results: str, extra: list[str]) -> Job:
    """Build a text-only baseline job.

    Args:
        method: ``closed_book``, ``rag`` or ``full_context``.
        source: Source config of the translated set; the full-context window follows its registered protocol.
        config: Dataset config.
        data_repo: Local data repository.
        results: Results root.
        extra: Further ``evaluate_baselines.py`` arguments.

    Returns:
        The job; results land in ``<results>/baselines/<method label>/<config>/``.
    """
    args = [
        "--method",
        method,
        "--config",
        config,
        "--scoring",
        CHOICE,
        "--data_repo",
        data_repo,
        "--max_new_tokens",
        str(MAX_NEW_TOKENS),
    ]
    label = method
    if method == "rag":
        args += ["--top_k", str(RAG_TOP_K)]
        label = f"rag_top{RAG_TOP_K}"
    if method == "full_context":
        protocol = BENCHMARKS[source].baseline
        args += ["--full_context_length", str(protocol.full_context_length)]
        if protocol.yarn_factor is not None:
            args += ["--yarn_factor", str(protocol.yarn_factor)]
        args += ["--max_num_seqs", str(FULL_CONTEXT_MAX_NUM_SEQS)]
    return Job(f"baselines/{label}/{config}", "baselines", f"{results}/baselines", tuple(args + extra))


def plan_jobs(
    sources: Iterable[str],
    tiers: Iterable[str],
    checkpoint_dir: str = DEFAULT_CHECKPOINTS,
    data_repo: str = DEFAULT_OUTPUT_REPO,
    results: str = DEFAULT_RESULTS,
    eval_args: list[str] | None = None,
    baseline_args: list[str] | None = None,
) -> list[Job]:
    """List the jobs of the evaluation plan, most important first.

    ``core`` runs each source's own checkpoint on the English and Turkish sets (Turkish with both the English and
    the Turkish system prompt) and the closed-book baseline on both. ``extended`` adds the "reply in Turkish" prompt,
    the cross-lingual sets, the general Stage 3 checkpoint and RAG. ``full`` adds the full-context baseline, which
    needs a 40 GB or larger GPU.

    Args:
        sources: Source configs of the translated sets (each must also name a checkpoint folder, e.g. ``pv4``).
        tiers: Tiers to include, from ``TIERS``.
        checkpoint_dir: Directory holding the downloaded checkpoint folders.
        data_repo: Local data repository written by ``tr_eval.build_data``.
        results: Results root.
        eval_args: Further arguments for every ``evaluate.py`` job.
        baseline_args: Further arguments for every ``evaluate_baselines.py`` job.

    Returns:
        The jobs.

    Raises:
        ValueError: If a tier is unknown or a source has no registered benchmark.
    """
    tiers = set(tiers)
    unknown = tiers - set(TIERS)
    if unknown:
        raise ValueError(f"Unknown tiers {sorted(unknown)}; choose from {TIERS}.")
    eval_args, baseline_args = list(eval_args or []), list(baseline_args or [])
    jobs = []
    for source in sources:
        if source not in BENCHMARKS:
            raise ValueError(f"{source!r} is not a registered benchmark; the plan needs its protocol.")
        en, tr, en2tr, tr2en = (f"{prefix}_{source}" for prefix in VARIANTS)
        evaluations, baselines = [], []  # (checkpoint, prompt, config) and (method, config)
        if "core" in tiers:
            evaluations += [(source, "en_prompt", en), (source, "en_prompt", tr), (source, "tr_prompt", tr)]
            baselines += [("closed_book", en), ("closed_book", tr)]
        if "extended" in tiers:
            evaluations += [
                (source, "en_prompt_reply_tr", tr),
                (source, "tr_prompt", en2tr),
                (source, "en_prompt", tr2en),
                (GENERAL_CHECKPOINT, "en_prompt", en),
                (GENERAL_CHECKPOINT, "en_prompt", tr),
            ]
            baselines += [("rag", en), ("rag", tr)]
        if "full" in tiers:
            baselines += [("full_context", en), ("full_context", tr)]
        jobs += [
            evaluate_job(checkpoint_dir, checkpoint, prompt, config, data_repo, results, eval_args)
            for checkpoint, prompt, config in evaluations
        ]
        jobs += [
            baseline_job(method, source, config, data_repo, results, baseline_args) for method, config in baselines
        ]
    return jobs


def format_jobs(jobs: Iterable[Job]) -> str:
    """Render jobs as a TSV table with a header.

    Args:
        jobs: The jobs.

    Returns:
        The table.
    """
    lines = ["\t".join(COLUMNS)]
    lines += ["\t".join((job.name, job.script, job.output_dir, shlex.join(job.args))) for job in jobs]
    return "\n".join(lines) + "\n"


def read_jobs(path: str | Path) -> list[Job]:
    """Read a TSV job table; blank lines and lines starting with ``#`` are ignored.

    Args:
        path: The table.

    Returns:
        The jobs, in table order.

    Raises:
        ValueError: If a line is malformed, names an unknown script, or repeats a job name.
    """
    jobs, names = [], set()
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#") or line.split("\t")[0] == COLUMNS[0]:
                continue
            fields = line.split("\t")
            if len(fields) != len(COLUMNS):
                raise ValueError(f"{path}:{number}: expected {len(COLUMNS)} tab-separated fields, got {len(fields)}.")
            name, script, output_dir, args = fields
            if script not in SCRIPTS:
                raise ValueError(f"{path}:{number}: unknown script {script!r}; expected one of {sorted(SCRIPTS)}.")
            if name in names:
                raise ValueError(f"{path}:{number}: duplicate job name {name!r}.")
            names.add(name)
            jobs.append(Job(name, script, output_dir, tuple(shlex.split(args))))
    return jobs


class QueueState:
    """Per-job state files under ``<results>/.queue/``: ``<job>.done`` (holding the command fingerprint),
    ``<job>.failed`` (exit code) and ``<job>.running`` (process id and start time)."""

    def __init__(self, results: Path) -> None:
        """Point at a results root.

        Args:
            results: The results root.
        """
        self.dir = results / STATE_DIR

    def _file(self, job: Job, state: str) -> Path:
        """Path of one state file.

        Args:
            job: The job.
            state: ``done``, ``failed`` or ``running``.

        Returns:
            The path.
        """
        return self.dir / f"{job.name.replace('/', '__')}.{state}"

    def state(self, job: Job) -> str:
        """Current state of a job.

        Args:
            job: The job.

        Returns:
            ``done`` (with the same command), ``running`` (its process is alive), ``failed`` or ``pending``.
        """
        done = self._file(job, DONE)
        if done.exists() and done.read_text().strip() == job.fingerprint():
            return DONE
        running = self._file(job, RUNNING)
        if running.exists():
            try:
                os.kill(json.loads(running.read_text())["pid"], 0)
                return RUNNING
            except (OSError, ValueError, KeyError):
                pass
        return FAILED if self._file(job, FAILED).exists() else PENDING

    def start(self, job: Job) -> None:
        """Mark a job as running in this process.

        Args:
            job: The job.
        """
        self.dir.mkdir(parents=True, exist_ok=True)
        self._file(job, RUNNING).write_text(json.dumps({"pid": os.getpid(), "started": time.time()}))

    def finish(self, job: Job, returncode: int) -> None:
        """Record a job's outcome.

        Args:
            job: The job.
            returncode: Exit code of its process.
        """
        self._file(job, RUNNING).unlink(missing_ok=True)
        if returncode == 0:
            self._file(job, FAILED).unlink(missing_ok=True)
            self._file(job, DONE).write_text(job.fingerprint() + "\n")
        else:
            self._file(job, DONE).unlink(missing_ok=True)
            self._file(job, FAILED).write_text(f"{returncode}\n")

    def started(self, job: Job) -> float | None:
        """When a running job started.

        Args:
            job: The job.

        Returns:
            The start time as a Unix timestamp, or None when the job is not marked as running.
        """
        try:
            return float(json.loads(self._file(job, RUNNING).read_text())["started"])
        except (OSError, ValueError, KeyError):
            return None


def sync_tree(source: Path, destination: Path) -> int:
    """Copy the files of ``source`` that are missing or differ in size or modification time under ``destination``.

    Running markers and the queue lock are not copied: they only mean something on the machine that wrote them.

    Args:
        source: Directory to copy from (nothing happens if it does not exist).
        destination: Directory to copy to.

    Returns:
        The number of files copied.
    """
    if not source.is_dir():
        return 0
    copied = 0
    for path in source.rglob("*"):
        if not path.is_file() or path.name == LOCK_FILE or path.suffix == f".{RUNNING}":
            continue
        target = destination / path.relative_to(source)
        stat = path.stat()
        if target.exists():
            existing = target.stat()
            if existing.st_size == stat.st_size and int(existing.st_mtime) >= int(stat.st_mtime):
                continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied += 1
    return copied


def sync_destination(directory: Path, target: Path) -> Path:
    """Where a directory is mirrored under a sync target: at its path relative to ``MEMORILLA_HOME`` when it lies
    inside it, so that copying the target back into ``MEMORILLA_HOME`` restores everything in place.

    Args:
        directory: A results, log or data directory.
        target: The sync target.

    Returns:
        The mirror directory.
    """
    try:
        return target / directory.resolve().relative_to(HOME.resolve())
    except ValueError:
        return target / directory.name


def run_jobs(
    jobs: list[Job],
    results: Path,
    log_dir: Path,
    python: str = sys.executable,
    sync: Callable[[], None] | None = None,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> dict[str, str]:
    """Run every job that is not done, in order, continuing past failures.

    Args:
        jobs: The jobs.
        results: Results root, which holds the queue state.
        log_dir: Directory of the job logs.
        python: Interpreter for the scripts.
        sync: Called after every job (e.g. to copy outputs to Google Drive), or None.
        runner: Runs a command; ``subprocess.run`` outside tests.

    Returns:
        The final state of every job by name.
    """
    state = QueueState(results)
    outcome = {}
    for job in jobs:
        if state.state(job) == DONE:
            outcome[job.name] = DONE
            continue
        log_path = log_dir / f"{job.name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{time.strftime('%H:%M:%S')}] start {job.name}", flush=True)
        state.start(job)
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {shlex.join(job.command(python))}\n")
            log.flush()
            returncode = runner(
                job.command(python), cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=False
            ).returncode
        state.finish(job, returncode)
        outcome[job.name] = DONE if returncode == 0 else FAILED
        print(f"[{time.strftime('%H:%M:%S')}] {outcome[job.name]} {job.name} (exit {returncode})", flush=True)
        if sync is not None:
            sync()
    return outcome


def status_table(jobs: list[Job], results: Path, now: float | None = None) -> str:
    """Summarise the state of every job.

    Args:
        jobs: The jobs.
        results: Results root, which holds the queue state.
        now: Current Unix time (None reads the clock).

    Returns:
        One line per job, then the totals.
    """
    state = QueueState(results)
    now = time.time() if now is None else now
    counts = {DONE: 0, FAILED: 0, RUNNING: 0, PENDING: 0}
    lines = []
    for job in jobs:
        current = state.state(job)
        counts[current] += 1
        detail = ""
        if current == RUNNING and (started := state.started(job)) is not None:
            detail = f"  ({(now - started) / 60:.0f} min)"
        lines.append(f"{current:<8} {job.name}{detail}")
    lines.append("  ".join(f"{name}: {count}" for name, count in counts.items()))
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="Print the job table.")
    plan.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES), help="Source configs.")
    plan.add_argument("--tiers", nargs="+", choices=TIERS, default=list(DEFAULT_TIERS), help="Tiers to include.")
    plan.add_argument("--checkpoints", default=DEFAULT_CHECKPOINTS, help="Directory of the checkpoint folders.")
    plan.add_argument("--data_repo", default=DEFAULT_OUTPUT_REPO, help="Local data repository.")
    plan.add_argument("--results", default=DEFAULT_RESULTS, help="Results root.")
    plan.add_argument(
        "--eval_args", default="", help="Extra evaluate.py arguments, e.g. '--gpu_memory_utilization 0.85'."
    )
    plan.add_argument("--baseline_args", default="", help="Extra evaluate_baselines.py arguments.")

    for name, help_text in (("run", "Run the pending jobs."), ("status", "Show the state of every job.")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("jobs", help="TSV job table written by 'plan'.")
        command.add_argument("--results", default=DEFAULT_RESULTS, help="Results root (holds the queue state).")
        command.add_argument("--only", default=None, help="Only jobs whose name matches this glob pattern.")
    run = commands.choices["run"]
    run.add_argument("--log_dir", default=DEFAULT_LOG_DIR, help="Directory of the job logs.")
    run.add_argument("--data_repo", default=DEFAULT_OUTPUT_REPO, help="Local data repository, synced with results.")
    run.add_argument("--sync_to", default=None, help="Copy results, logs and data repository here after every job.")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Dispatch a subcommand.

    Args:
        args: Parsed command-line arguments.

    Raises:
        SystemExit: If another ``run`` already holds the queue lock, or some jobs failed.
    """
    if args.command == "plan":
        jobs = plan_jobs(
            args.sources,
            args.tiers,
            args.checkpoints,
            args.data_repo,
            args.results,
            shlex.split(args.eval_args),
            shlex.split(args.baseline_args),
        )
        sys.stdout.write(format_jobs(jobs))
        return

    jobs = [job for job in read_jobs(args.jobs) if args.only is None or fnmatch(job.name, args.only)]
    results = resolve(args.results)
    if args.command == "status":
        print(status_table(jobs, results))
        return

    log_dir, data_repo = resolve(args.log_dir), resolve(args.data_repo)
    sync = None
    if args.sync_to:
        target = Path(args.sync_to)

        def sync() -> None:
            """Mirror the results, logs and data repository under the sync target."""
            for directory in (results, log_dir, data_repo):
                sync_tree(directory, sync_destination(directory, target))

    results.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(results / LOCK_FILE), timeout=0):
            outcome = run_jobs(jobs, results, log_dir, sync=sync)
    except Timeout:
        raise SystemExit(f"Another queue is already running on {results}; check it with 'status'.") from None
    failed = [name for name, state in outcome.items() if state == FAILED]
    if failed:
        raise SystemExit(f"{len(failed)} jobs failed: {', '.join(failed)}. See their logs; 'run' again retries them.")


if __name__ == "__main__":
    main(parse_args())
