"""Tests for the Turkish evaluation: data building with a fake translator, the run queue and the Turkish scores."""

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any

import pytest

from evaluate_baselines import parse_args as parse_baseline_args
from evaluate_baselines import resolve_targets as resolve_baseline_targets
from memorilla.benchmarks import CHOICE, CUSTOM_FULL_CONTEXT_LENGTH, HELPFUL_PROMPT, custom_benchmark
from memorilla.data import load_split
from memorilla.metrics import check_scoring_inputs, fit_choice_metrics
from tr_eval.build_data import (
    MANIFEST_FILE,
    VARIANTS,
    Translator,
    build_variants,
    parse_qa,
    sample_rows,
    translate_rows,
    write_config,
)
from tr_eval.common import JsonlCache, parse_json_object
from tr_eval.run_queue import (
    DONE,
    FAILED,
    PENDING,
    Job,
    QueueState,
    format_jobs,
    plan_jobs,
    read_jobs,
    run_jobs,
    status_table,
    sync_tree,
)
from tr_eval.score import Judge, judge_indices, judge_run, language_metrics, parse_judgement, write_summary

PAYLOAD_MARKER = "Input:\n"
TRANSLATED = "TR "


class FakeTranslatorResponses:
    """Stands in for ``client.responses``: "translates" by prefixing ``TR``, and can fail the first replies."""

    def __init__(self, bad_replies: int = 0) -> None:
        """Set up the fake.

        Args:
            bad_replies: Number of initial question replies that are not valid JSON.
        """
        self.calls = 0
        self.bad_replies = bad_replies

    def create(self, model: str, input: list[dict[str, str]], reasoning: dict[str, str]) -> SimpleNamespace:
        """Answer one prompt.

        Args:
            model: Model name (ignored).
            input: The chat input; its single user message is the prompt.
            reasoning: Reasoning settings (ignored).

        Returns:
            An object with ``output_text``.
        """
        self.calls += 1
        prompt = input[0]["content"]
        if PAYLOAD_MARKER not in prompt:
            return SimpleNamespace(output_text=TRANSLATED + prompt.rsplit("Text:\n", 1)[1])
        if self.bad_replies > 0:
            self.bad_replies -= 1
            return SimpleNamespace(output_text="not json")
        payload = json.loads(prompt.rsplit(PAYLOAD_MARKER, 1)[1])
        reply = {
            "question": TRANSLATED + payload["question"],
            "options": [TRANSLATED + option for option in payload["options"]],
        }
        return SimpleNamespace(output_text=f"```json\n{json.dumps(reply)}\n```")


def make_translator(cache_path: Path, responses: FakeTranslatorResponses, max_retries: int = 2) -> Translator:
    """Build a translator around a fake client.

    Args:
        cache_path: Translation cache file.
        responses: The fake ``client.responses``.
        max_retries: Attempts per request.

    Returns:
        The translator.
    """
    return Translator(SimpleNamespace(responses=responses), "fake", "none", max_retries, JsonlCache(cache_path))


def toy_rows(data_repo: Path) -> list[dict[str, Any]]:
    """Load the toy training split of the fixture repository.

    Args:
        data_repo: The fixture repository.

    Returns:
        Its rows.
    """
    return load_split("toy", "train", data_repo).to_list()


def test_sample_rows_keeps_whole_collections() -> None:
    """Sampling is deterministic, keeps stored order, and samples collections before questions."""
    collections = [1, 2, 1, 3, 2, 1]
    assert sample_rows(collections, 0, 0, seed=0) == list(range(6))
    picked = sample_rows(collections, 2, 0, seed=5)
    assert picked == sample_rows(collections, 2, 0, seed=5) == sorted(picked)
    kept = {collections[index] for index in picked}
    assert len(kept) == 2 and picked == [index for index in range(6) if collections[index] in kept]
    assert len(sample_rows(collections, 0, 1, seed=0)) == 3


def test_parse_qa_checks_the_translation() -> None:
    """A question translation must keep the number of options and keep distinct options distinct."""
    reply = '```json\n{"question": " Soru ", "options": [" A ", "B"]}\n```'
    assert parse_qa(reply, ["a", "b"]) == ("Soru", ["A", "B"])
    for bad in ('{"question": "Soru", "options": ["A"]}', '{"question": "Soru", "options": ["A", "A"]}', "[1]"):
        with pytest.raises(ValueError):
            parse_qa(bad, ["a", "b"])
    with pytest.raises(ValueError):
        parse_qa('{"question": "", "options": ["A", "B"]}', ["a", "b"])
    assert parse_qa('{"question": "S", "options": ["A", "A"]}', ["x", "x"]) == ("S", ["A", "A"])


def test_build_writes_matched_variants(data_repo: Path, tmp_path: Path) -> None:
    """The four configs share their rows; Turkish answers stay among the Turkish choices; reruns hit the cache."""
    rows = toy_rows(data_repo)
    responses = FakeTranslatorResponses()
    translator = make_translator(tmp_path / "cache.jsonl", responses)
    documents, qa, failed = translate_rows(translator, rows, max_workers=4)
    assert not failed
    distinct = {document for row in rows for document in row["documents"]}
    assert responses.calls == len(distinct) + len(rows)

    variants = build_variants(rows, documents, qa, set())
    output, manifest = tmp_path / "out", {}
    for prefix, variant_rows in variants.items():
        assert len(variant_rows) == len(rows)
        assert write_config(output, f"{prefix}_toy", "test", variant_rows, manifest)
    tr = load_split("tr_toy", "test", output).to_list()
    en2tr = load_split("xlang_en2tr_toy", "test", output).to_list()
    tr2en = load_split("xlang_tr2en_toy", "test", output).to_list()
    for source, turkish, mixed_en_docs, mixed_tr_docs in zip(rows, tr, en2tr, tr2en, strict=True):
        assert turkish["collection_id"] == source["collection_id"] and turkish["hard"] == source["hard"]
        assert turkish["question"] == TRANSLATED + source["question"].strip()
        assert turkish["answer"] == TRANSLATED + source["answer"] and turkish["answer"] in turkish["choices"]
        assert [bool(choice) for choice in turkish["choices"]] == [bool(choice) for choice in source["choices"]]
        assert turkish["documents"] == [TRANSLATED + document for document in source["documents"]]
        assert mixed_en_docs["documents"] == source["documents"] and mixed_en_docs["question"] == turkish["question"]
        assert mixed_tr_docs["documents"] == turkish["documents"] and mixed_tr_docs["question"] == source["question"]
    dataset = load_split("tr_toy", "test", output)
    check_scoring_inputs(fit_choice_metrics(custom_benchmark("tr_toy", CHOICE), dataset), dataset)

    stale = output / "embeddings" / "enc" / "tr_toy" / "test"
    stale.mkdir(parents=True)
    assert not write_config(output, "tr_toy", "test", variants["tr"], manifest)
    assert stale.exists()
    assert write_config(output, "tr_toy", "test", variants["tr"][:1], manifest)
    assert not stale.exists()

    rerun = FakeTranslatorResponses()
    translate_rows(make_translator(tmp_path / "cache.jsonl", rerun), rows, max_workers=4)
    assert rerun.calls == 0
    assert set(VARIANTS) == {"en", "tr", "xlang_en2tr", "xlang_tr2en"} and MANIFEST_FILE


def test_bad_question_translations_are_retried_then_reported(data_repo: Path, tmp_path: Path) -> None:
    """Unparsable replies are retried; a question that never parses fails its row only."""
    rows = toy_rows(data_repo)[:1]
    translator = make_translator(tmp_path / "retry.jsonl", FakeTranslatorResponses(bad_replies=1))
    _, qa, failed = translate_rows(translator, rows, max_workers=1)
    assert not failed and qa[0].question.startswith(TRANSLATED)

    translator = make_translator(tmp_path / "fail.jsonl", FakeTranslatorResponses(bad_replies=5), max_retries=2)
    documents, qa, failed = translate_rows(translator, rows, max_workers=1)
    assert failed == {0} and not qa and documents
    assert build_variants(rows, documents, qa, failed) == {prefix: [] for prefix in VARIANTS}


def test_jsonl_cache_survives_a_truncated_line(tmp_path: Path) -> None:
    """A partial last line is skipped on reading, and the next entry starts on a new line."""
    path = tmp_path / "cache.jsonl"
    path.write_text('{"key": "a", "value": 1}\n{"key": "b", "val')
    cache = JsonlCache(path)
    assert "a" in cache and "b" not in cache and len(cache) == 1
    cache.put("c", {"x": "ğüşiöç"})
    reread = JsonlCache(path)
    assert reread.get("c") == {"x": "ğüşiöç"} and reread.get("a") == 1
    assert parse_json_object('```\n{"a": 1}\n```') == {"a": 1}


def test_plan_round_trip() -> None:
    """The plan lists the core jobs first and survives the TSV round trip, Turkish prompts included."""
    jobs = plan_jobs(["pv4"], ["core", "full"], eval_args=["--gpu_memory_utilization", "0.85"])
    names = [job.name for job in jobs]
    assert names[:3] == ["pv4/en_prompt/en_pv4", "pv4/en_prompt/tr_pv4", "pv4/tr_prompt/tr_pv4"]
    assert "baselines/closed_book/tr_pv4" in names and "baselines/full_context/tr_pv4" in names
    full = next(job for job in jobs if job.name == "baselines/full_context/en_pv4")
    assert "--yarn_factor" in full.args and "65536" in full.args
    assert all(job.args[-2:] == ("--gpu_memory_utilization", "0.85") for job in jobs if job.script == "evaluate")
    with pytest.raises(ValueError):
        plan_jobs(["toy"], ["core"])
    with pytest.raises(ValueError):
        plan_jobs(["pv4"], ["everything"])


def test_read_jobs_round_trip(tmp_path: Path) -> None:
    """A written plan reads back unchanged; malformed tables are rejected."""
    jobs = plan_jobs(["pmv2"], ["core", "extended"])
    path = tmp_path / "runs.tsv"
    path.write_text("# comment\n\n" + format_jobs(jobs))
    assert read_jobs(path) == jobs
    path.write_text("a\tevaluate\tout\t\na\tevaluate\tout\t\n")
    with pytest.raises(ValueError):
        read_jobs(path)
    path.write_text("a\tunknown\tout\t\n")
    with pytest.raises(ValueError):
        read_jobs(path)


def test_run_jobs_resumes(tmp_path: Path) -> None:
    """Failed jobs are retried, done jobs skipped, and a changed command reruns a done job."""
    results, logs = tmp_path / "results", tmp_path / "logs"
    jobs = [Job("a/one", "evaluate", "out/a", ("--x", "1")), Job("b/two", "baselines", "out/b", ())]
    calls: list[list[str]] = []
    exit_codes = iter([1, 0, 0])

    def runner(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        """Record a command and return the next exit code.

        Args:
            command: The command.
            **kwargs: ``subprocess.run`` options (ignored).

        Returns:
            The completed process.
        """
        calls.append(command)
        return subprocess.CompletedProcess(command, next(exit_codes))

    synced = []
    outcome = run_jobs(jobs, results, logs, python="py", sync=lambda: synced.append(1), runner=runner)
    assert outcome == {"a/one": FAILED, "b/two": DONE} and len(synced) == 2
    assert calls[0] == ["py", "evaluate.py", "--x", "1", "--output_dir", "out/a"]
    assert (logs / "a" / "one.log").read_text().count("===") == 1
    assert "failed   a/one" in status_table(jobs, results)

    assert run_jobs(jobs, results, logs, python="py", runner=runner) == {"a/one": DONE, "b/two": DONE}
    assert len(calls) == 3
    changed = Job("b/two", "baselines", "out/b", ("--top_k", "3"))
    assert QueueState(results).state(changed) == PENDING


def test_sync_tree_copies_changes_only(tmp_path: Path) -> None:
    """Only new or changed files are copied, and running markers stay behind."""
    source, target = tmp_path / "src", tmp_path / "dst"
    (source / "run").mkdir(parents=True)
    (source / "run" / "metrics.json").write_text("{}")
    (source / "job.running").write_text("{}")
    assert sync_tree(source, target) == 1
    assert sync_tree(source, target) == 0
    assert not (target / "job.running").exists()
    assert sync_tree(tmp_path / "missing", target) == 0


def test_baselines_accept_a_config() -> None:
    """``evaluate_baselines.py --config`` builds a generic protocol; overrides apply to it."""
    args = parse_baseline_args(
        ["--config", "tr_pv4", "--method", "full_context", "--scoring", CHOICE, "--yarn_factor", "2.0"]
        + ["--full_context_length", "65536", "--max_new_tokens", "384"]
    )
    (benchmark,) = resolve_baseline_targets(args)
    assert benchmark.name == "tr_pv4" and benchmark.scoring == CHOICE
    assert benchmark.baseline.system_prompt == HELPFUL_PROMPT and benchmark.baseline.max_new_tokens == 384
    assert benchmark.baseline.full_context_length == 65536 and benchmark.baseline.yarn_factor == 2.0
    (benchmark,) = resolve_baseline_targets(parse_baseline_args(["--config", "x", "--method", "rag"]))
    assert benchmark.baseline.full_context_length == CUSTOM_FULL_CONTEXT_LENGTH
    (registered,) = resolve_baseline_targets(
        parse_baseline_args(["--benchmark", "pv4", "--method", "rag", "--system_prompt", "Merhaba."])
    )
    assert registered.baseline.system_prompt == "Merhaba." and registered.baseline.yarn_factor == 2.0
    with pytest.raises(SystemExit):
        parse_baseline_args(["--benchmark", "pv4", "--method", "rag", "--yarn_factor", "2.0"])


def fake_detect(text: str) -> tuple[str | None, list[tuple[str, int]]]:
    """Detect Turkish by its letters, and English words in brackets as an English segment.

    Args:
        text: The text.

    Returns:
        The language and the segments.
    """
    if not text.strip():
        return None, []
    language = "tr" if any(letter in text for letter in "çğışöü") else "en"
    segments = [(language, len(text.split()))]
    if "[" in text:
        segments.append(("en", len(text[text.index("[") :].split())))
    return language, segments


def test_language_metrics() -> None:
    """Answers are checked against the language of their question, and mixed-in English runs are counted."""
    questions = ["Kahvaltıda ne yiyeyim?", "Hangi şehre gideyim?", "What should I cook?", "Ne içeyim?"]
    predictions = ["Yulaf ezmesi öneririm.", "Paris", "Köfte yap.", "Çay iç [with lemon and honey]"]
    scores = language_metrics(questions, predictions, fake_detect)
    assert scores == {"language_fidelity": 0.5, "turkish": 0.75, "mixed": 0.25}


def test_lingua_detector() -> None:
    """The real detector tells Turkish from English and finds an English run inside a Turkish answer."""
    pytest.importorskip("lingua")
    from tr_eval.score import make_language_detector

    detect = make_language_detector()
    assert detect("Sabah kahvaltısında yumurta ve avokado deneyebilirsin.")[0] == "tr"
    assert detect("Since you love hiking, try a hearty oatmeal.")[0] == "en"
    scores = language_metrics(
        ["Kahvaltıda ne yemeliyim?"],
        ["Bence oatmeal with berries and chia seeds harika olur, çünkü sabahları uzun süre tok kalıyorsun."],
        detect,
    )
    assert scores["mixed"] == 1.0


def test_chrf_handles_turkish() -> None:
    """chrF++ rewards shared stems that whitespace-token metrics miss."""
    pytest.importorskip("sacrebleu")
    from tr_eval.score import chrf

    assert chrf(["güzel bir gün"], ["güzel bir gün"]) == pytest.approx(1.0)
    assert 0.3 < chrf(["evlerimizde"], ["evlerimiz"]) < 1.0


def test_judge(tmp_path: Path) -> None:
    """Judge replies are validated, averaged over a fixed sample, and cached by content."""
    assert (
        parse_judgement(
            '{"personalization": 5, "consistency": 4.0, "no_fabrication": 5, "grammar": 3, '
            '"naturalness": 2, "address": 1, "language": "TR"}'
        )["language"]
        == "tr"
    )
    for bad in ('{"personalization": 6}', '{"personalization": true}', "nope"):
        with pytest.raises(ValueError):
            parse_judgement(bad)
    assert judge_indices(10, 3, seed=1) == judge_indices(10, 3, seed=1) and judge_indices(2, 5, seed=1) == [0, 1]

    scores = {"personalization": 4, "consistency": 3, "no_fabrication": 5, "grammar": 5, "naturalness": 4}
    replies = iter(["oops", json.dumps(scores | {"address": 2, "language": "tr"})])
    responses = SimpleNamespace(calls=0)

    def create(**kwargs: Any) -> SimpleNamespace:
        """Return the next reply.

        Args:
            **kwargs: Request options (ignored).

        Returns:
            An object with ``output_text``.
        """
        responses.calls += 1
        return SimpleNamespace(output_text=next(replies))

    judge = Judge(SimpleNamespace(responses=SimpleNamespace(create=create)), "m", "none", 2, JsonlCache(tmp_path / "j"))
    records = [{"question": "Soru?", "answer": "Cevap", "prediction": "Yanıt"}] * 2
    means, rows = judge_run(judge, records, [0, 1], max_workers=1)
    assert means["judge_n"] == 2 and means["judge_personalization"] == 4.0 and means["judge_address"] == 2.0
    assert [row["row"] for row in rows] == [0, 1] and responses.calls == 2

    write_summary([{"run": "a", "accuracy": 50.0}, {"run": "b", "chrf": 12.5}], tmp_path)
    assert (tmp_path / "summary.md").read_text().splitlines()[0] == "| run | accuracy | chrf |"
