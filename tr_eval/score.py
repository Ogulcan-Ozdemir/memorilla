"""Turkish-aware scores for the outputs of ``evaluate.py`` and ``evaluate_baselines.py``.

The repository's generation metrics are unreliable for Turkish: ``rouge_score``'s default tokenizer replaces every
character outside ``[a-z0-9]`` with a space (so "güzel" becomes "g zel") and stems in English, METEOR matches with
English WordNet, ``str.lower`` maps "I" to "i" instead of "ı", and word-overlap metrics punish Turkish suffixes. This
script adds, for every run under a results root (every directory with a ``predictions.jsonl``):

- language fidelity: the share of answers in the question's language, the share in Turkish, and the share that mix
  in a run of at least ``MIXED_MIN_WORDS`` words of another language (English and Turkish only);
- chrF++ against the reference answer, which compares character n-grams and so tolerates suffixes;
- optionally, the mean cosine similarity of answer and reference under the encoder (needs a GPU);
- optionally, an LLM judge on a fixed sample of rows (the same rows for every run of a config, so runs are paired).

It writes ``tr_metrics.json`` next to each run's ``metrics.json``, a ``summary.csv`` and ``summary.md`` over all runs,
and optionally a blind sheet for human raters. Run from the repository root:
``python -m tr_eval.score --results results/tr --judge_samples 40``.
"""

import argparse
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
import csv
from fnmatch import fnmatch
import json
import logging
from pathlib import Path
import random
from typing import TYPE_CHECKING, Any

import numpy as np

from memorilla.metrics import METRICS_FILE, PREDICTIONS_FILE
from memorilla.paths import DEFAULT_ENCODER, MODEL_CACHE_DIR, resolve
from memorilla.utils import DEFAULT_SEED, write_json
from tr_eval.build_data import TURKISH, VARIANTS
from tr_eval.common import (
    USAGE,
    BudgetExceeded,
    JsonlCache,
    complete,
    create_client,
    log_key_usage,
    parse_json_object,
    set_budget,
    stable_key,
)
from tr_eval.prompts import judge_prompt
from tr_eval.run_queue import DEFAULT_RESULTS, STATE_DIR

if TYPE_CHECKING:
    from openai import OpenAI
    from vllm import LLM

LOGGER = logging.getLogger(__name__)

TR_METRICS_FILE = "tr_metrics.json"
JUDGE_FILE = "judge.jsonl"
JUDGE_CACHE_FILE = "judge_cache.jsonl"
SUMMARY_CSV = "summary.csv"
SUMMARY_MD = "summary.md"
HUMAN_SHEET = "human_ratings.csv"
HUMAN_KEY = "human_key.csv"
# A short or uncertain segment is usually a misreading (names such as "Leroy Jenkins'in", or "may ask for a " read as
# Turkish), and long answers have many of them, so only confident runs of several words count as mixing.
MIXED_MIN_WORDS = 5
MIXED_MIN_CONFIDENCE = 0.9
JUDGE_CRITERIA = ("personalization", "consistency", "no_fabrication", "grammar", "naturalness", "address")
JUDGE_SCALE = (1, 5)
DEFAULT_JUDGE_MODEL = "anthropic/claude-haiku-5.5"
DEFAULT_JUDGE_REASONING_EFFORT = "medium"
DEFAULT_MAX_WORKERS = 16
DEFAULT_MAX_RETRIES = 3
SIMILARITY_GPU_MEMORY_UTILIZATION = 0.9
TURKISH_QUESTION_PREFIXES = tuple(f"{prefix}_" for prefix, (_, qa) in VARIANTS.items() if qa == TURKISH)

Detection = tuple[str | None, list[tuple[str, int]]]


def make_language_detector() -> Callable[[str], Detection]:
    """Build an English/Turkish language detector.

    Returns:
        A function from a text to its language (``en``, ``tr`` or None when undetermined) and its language segments
        as ``(language, word count)`` pairs, keeping only segments whose language the detector, run on the segment
        alone, gives a confidence of at least ``MIXED_MIN_CONFIDENCE``.

    Raises:
        ImportError: If ``lingua-language-detector`` is not installed.
    """
    try:
        from lingua import Language, LanguageDetectorBuilder
    except ImportError as error:
        raise ImportError("Language detection needs lingua: pip install -e '.[tr]'") from error
    detector = LanguageDetectorBuilder.from_languages(Language.ENGLISH, Language.TURKISH).build()

    def detect(text: str) -> Detection:
        """Detect the language of a text and of its segments.

        Args:
            text: The text.

        Returns:
            The language and the segments.
        """
        language = detector.detect_language_of(text)
        segments = [
            (result.language.iso_code_639_1.name.lower(), result.word_count)
            for result in detector.detect_multiple_languages_of(text)
            if detector.compute_language_confidence(text[result.start_index : result.end_index], result.language)
            >= MIXED_MIN_CONFIDENCE
        ]
        return (language.iso_code_639_1.name.lower() if language else None), segments

    return detect


def language_metrics(
    questions: Sequence[str], predictions: Sequence[str], detect: Callable[[str], Detection]
) -> dict[str, float]:
    """Measure whether answers are in the language of their question.

    Args:
        questions: The questions.
        predictions: One answer per question.
        detect: Language detector from ``make_language_detector``.

    Returns:
        As fractions: ``language_fidelity`` (answer language equals question language, over questions whose language
        was determined), ``turkish`` (answers in Turkish) and ``mixed`` (answers containing a run of at least
        ``MIXED_MIN_WORDS`` words in another language than the answer's).
    """
    matched, determined, turkish, mixed = 0, 0, 0, 0
    for question, prediction in zip(questions, predictions, strict=True):
        expected, _ = detect(question)
        language, segments = detect(prediction)
        if expected is not None:
            determined += 1
            matched += language == expected
        turkish += language == TURKISH
        mixed += language is not None and any(
            other != language and words >= MIXED_MIN_WORDS for other, words in segments
        )
    count = max(len(predictions), 1)
    return {
        "language_fidelity": matched / max(determined, 1),
        "turkish": turkish / count,
        "mixed": mixed / count,
    }


def chrf(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Corpus chrF++ (character n-grams plus word bigrams).

    Args:
        predictions: Generated texts.
        references: One reference per prediction.

    Returns:
        The score as a fraction.

    Raises:
        ImportError: If ``sacrebleu`` is not installed.
    """
    try:
        from sacrebleu.metrics import CHRF
    except ImportError as error:
        raise ImportError("chrF++ needs sacrebleu: pip install -e '.[tr]'") from error
    return CHRF(word_order=2).corpus_score(list(predictions), [list(references)]).score / 100


def embedding_similarity(predictions: Sequence[str], references: Sequence[str], engine: "LLM") -> float:
    """Mean cosine similarity of each prediction and its reference under the encoder.

    Args:
        predictions: Generated texts.
        references: One reference per prediction.
        engine: Embedding engine from ``memorilla.embeddings.load_encoder``.

    Returns:
        The mean similarity.
    """
    from memorilla.embeddings import embed_texts

    vectors = embed_texts(engine, [*predictions, *references]).astype(np.float32)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    count = len(predictions)
    return float(np.mean(np.sum(vectors[:count] * vectors[count:], axis=1)))


def judge_indices(num_rows: int, size: int, seed: int) -> list[int]:
    """Pick the rows to judge; every run with the same number of rows (i.e. of the same config) gets the same rows.

    Args:
        num_rows: Rows of the run.
        size: Rows to pick; 0 picks none.
        seed: Sampling seed.

    Returns:
        The sorted row indices.
    """
    return sorted(random.Random(seed).sample(range(num_rows), min(size, num_rows)))


def parse_judgement(reply: str) -> dict[str, Any]:
    """Parse and check a judge reply.

    Args:
        reply: The judge's reply.

    Returns:
        The integer score of every criterion and the ``language`` the judge reported.

    Raises:
        ValueError: If the reply is not a JSON object or a score is missing or outside ``JUDGE_SCALE``.
    """
    value = parse_json_object(reply)
    low, high = JUDGE_SCALE
    scores: dict[str, Any] = {}
    for criterion in JUDGE_CRITERIA:
        score = value.get(criterion)
        if isinstance(score, float) and score.is_integer():
            score = int(score)
        if not isinstance(score, int) or isinstance(score, bool) or not low <= score <= high:
            raise ValueError(f"Bad {criterion!r} score: {score!r}.")
        scores[criterion] = score
    scores["language"] = str(value.get("language", "")).strip().lower()
    return scores


class Judge:
    """Rates answers with an LLM, through a cache keyed by the judged content."""

    def __init__(
        self, client: "OpenAI", model: str, reasoning_effort: str, max_retries: int, cache: JsonlCache
    ) -> None:
        """Set up the judge.

        Args:
            client: Client from ``tr_eval.common.create_client`` (or a test double; see ``tr_eval.common.complete``).
            model: Judge model.
            reasoning_effort: Reasoning effort for ``model``.
            max_retries: Attempts per request, and per unparsable reply.
            cache: Cache of finished judgements.
        """
        self.client = client
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_retries = max_retries
        self.cache = cache

    def __call__(self, question: str, reference: str, prediction: str) -> dict[str, Any] | None:
        """Judge one answer.

        Args:
            question: The question.
            reference: The reference answer.
            prediction: The answer to judge.

        Returns:
            The scores, or None when no reply could be parsed within ``max_retries`` attempts.
        """
        key = stable_key("judge", self.model, question, reference, prediction)
        if key not in self.cache:
            prompt = judge_prompt(question, reference, prediction)
            for attempt in range(self.max_retries):
                try:
                    reply = complete(self.client, prompt, self.model, self.reasoning_effort, self.max_retries)
                    self.cache.put(key, parse_judgement(reply))
                    break
                except BudgetExceeded:
                    raise
                except (ValueError, RuntimeError) as error:
                    LOGGER.warning("Judge failed (attempt %d/%d): %s", attempt + 1, self.max_retries, error)
            else:
                return None
        return self.cache.get(key)


def judge_run(
    judge: Judge, records: list[dict[str, Any]], indices: list[int], max_workers: int
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Judge the sampled rows of one run.

    Args:
        judge: The judge.
        records: The run's prediction records.
        indices: Rows to judge.
        max_workers: Concurrent requests.

    Returns:
        The mean score of every criterion (as ``judge_<criterion>``) with the number of judged rows
        (``judge_n``), and the per-row judgements.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(
            pool.map(
                lambda index: judge(
                    str(records[index]["question"]), str(records[index]["answer"]), str(records[index]["prediction"])
                ),
                indices,
            )
        )
    rows = [{"row": index, **result} for index, result in zip(indices, results, strict=True) if result is not None]
    if len(rows) < len(indices):
        LOGGER.warning("%d of %d judgements failed.", len(indices) - len(rows), len(indices))
    means: dict[str, float] = {"judge_n": len(rows)}
    for criterion in JUDGE_CRITERIA:
        means[f"judge_{criterion}"] = round(float(np.mean([row[criterion] for row in rows])), 2) if rows else 0.0
    return means, rows


def find_runs(results: Path) -> list[Path]:
    """Find every run directory under a results root.

    Args:
        results: The results root.

    Returns:
        The directories holding a ``predictions.jsonl``, sorted.
    """
    return sorted(path.parent for path in results.rglob(PREDICTIONS_FILE) if STATE_DIR not in path.parts)


def read_records(run: Path) -> list[dict[str, Any]]:
    """Read a run's prediction records.

    Args:
        run: The run directory.

    Returns:
        The records, in row order.
    """
    with open(run / PREDICTIONS_FILE, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object, or return an empty one when the file does not exist.

    Args:
        path: The file.

    Returns:
        The object.
    """
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def percent(value: float) -> float:
    """Express a fraction as a percentage with two decimals, as ``metrics.json`` does.

    Args:
        value: The fraction.

    Returns:
        The percentage.
    """
    return round(100 * value, 2)


def write_summary(rows: list[dict[str, Any]], results: Path) -> None:
    """Write the summary over all runs as CSV and as a Markdown table.

    Args:
        rows: One row per run, starting with its ``run`` name.
        results: The results root.
    """
    columns = list(dict.fromkeys(column for row in rows for column in row))
    with open(results / SUMMARY_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(str(row.get(column, "")) for column in columns) + " |" for row in rows]
    (results / SUMMARY_MD).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_human_sheet(items: list[dict[str, Any]], results: Path, seed: int) -> None:
    """Write a blind, shuffled rating sheet and the key that maps its items back to runs and rows.

    Args:
        items: ``run``, ``row``, ``question``, ``reference`` and ``prediction`` of every item to rate.
        results: The results root.
        seed: Shuffle seed.
    """
    items = list(items)
    random.Random(seed).shuffle(items)
    with open(results / HUMAN_SHEET, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item", "question", "reference", "prediction", *JUDGE_CRITERIA, "notes"])
        for number, item in enumerate(items, start=1):
            writer.writerow([number, item["question"], item["reference"], item["prediction"]])
    with open(results / HUMAN_KEY, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item", "run", "row"])
        for number, item in enumerate(items, start=1):
            writer.writerow([number, item["run"], item["row"]])


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=DEFAULT_RESULTS, help="Results root to score.")
    parser.add_argument("--judge_samples", type=int, default=0, help="Rows judged per run; 0 skips the judge.")
    parser.add_argument("--judge_model", default=DEFAULT_JUDGE_MODEL, help="Judge model.")
    parser.add_argument(
        "--judge_reasoning_effort", default=DEFAULT_JUDGE_REASONING_EFFORT, help="Its reasoning effort."
    )
    parser.add_argument(
        "--judge_include",
        nargs="+",
        default=["*"],
        help="Judge only runs whose name (relative to --results) matches one of these glob patterns, e.g. '*/tr_*'.",
    )
    parser.add_argument(
        "--max_cost",
        type=float,
        default=None,
        help="Stop sending OpenRouter judge requests once this run has spent this many USD (judgements stay cached).",
    )
    parser.add_argument("--max_workers", type=int, default=DEFAULT_MAX_WORKERS, help="Concurrent judge requests.")
    parser.add_argument("--max_retries", type=int, default=DEFAULT_MAX_RETRIES, help="Attempts per judge request.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Seed of the judge sample and human sheet.")
    parser.add_argument(
        "--embedding_similarity", action="store_true", help="Also compute answer-reference similarity (GPU)."
    )
    parser.add_argument("--encoder", default=DEFAULT_ENCODER, help="Encoder for --embedding_similarity.")
    parser.add_argument("--cache_dir", default=MODEL_CACHE_DIR, help="Model cache directory.")
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=SIMILARITY_GPU_MEMORY_UTILIZATION,
        help="Fraction of GPU memory for the encoder.",
    )
    parser.add_argument(
        "--human_sheet",
        type=int,
        default=0,
        help="Items per Turkish-question run in a blind human rating sheet, taken from the judge sample; 0 skips it.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Score every run under the results root and write the summaries.

    Args:
        args: Parsed command-line arguments.

    Raises:
        SystemExit: If the results root holds no runs.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results = resolve(args.results)
    runs = find_runs(results)
    if not runs:
        raise SystemExit(f"No {PREDICTIONS_FILE} under {results}.")
    detect = make_language_detector()
    judge = None
    if args.judge_samples > 0:
        cache = JsonlCache(results / JUDGE_CACHE_FILE)
        judge = Judge(create_client(), args.judge_model, args.judge_reasoning_effort, args.max_retries, cache)
        set_budget(args.max_cost)
        log_key_usage()
    engine = None
    if args.embedding_similarity:
        from memorilla.embeddings import load_encoder

        engine = load_encoder(
            args.encoder, cache_dir=args.cache_dir, gpu_memory_utilization=args.gpu_memory_utilization
        )

    summary, human_items = [], []
    for run in runs:
        name = run.relative_to(results).as_posix()
        records = read_records(run)
        questions = [str(record["question"]) for record in records]
        predictions = [str(record["prediction"]) for record in records]
        references = [str(record["answer"]) for record in records]
        scores = read_json(run / TR_METRICS_FILE)
        scores.update({key: percent(value) for key, value in language_metrics(questions, predictions, detect).items()})
        scores["chrf"] = percent(chrf(predictions, references))
        if engine is not None:
            scores["embedding_similarity"] = round(embedding_similarity(predictions, references, engine), 4)
        indices = judge_indices(len(records), args.judge_samples, args.seed)
        if judge is not None and any(fnmatch(name, pattern) for pattern in args.judge_include):
            means, judged = judge_run(judge, records, indices, args.max_workers)
            scores.update(means)
            with open(run / JUDGE_FILE, "w", encoding="utf-8") as handle:
                handle.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in judged)
        write_json(run / TR_METRICS_FILE, scores)
        summary.append({"run": name, **read_json(run / METRICS_FILE), **scores})
        LOGGER.info("%s: %s", name, json.dumps(scores, ensure_ascii=False))
        if args.human_sheet > 0 and run.name.startswith(TURKISH_QUESTION_PREFIXES):
            pool = indices if indices else list(range(len(records)))
            human_items += [
                {"run": name, "row": index, "question": questions[index], "reference": references[index]}
                | {"prediction": predictions[index]}
                for index in sorted(random.Random(args.seed).sample(pool, min(args.human_sheet, len(pool))))
            ]
    write_summary(summary, results)
    if human_items:
        write_human_sheet(human_items, results, args.seed)
    LOGGER.info("Wrote %s and %s under %s.", SUMMARY_CSV, SUMMARY_MD, results)
    if judge is not None:
        LOGGER.info("Judge API usage (new requests only; cached judgements are free): %s", USAGE.summary())
        log_key_usage()


if __name__ == "__main__":
    main(parse_args())
