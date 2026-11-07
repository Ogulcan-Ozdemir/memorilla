"""Build Turkish test sets from a Memorilla-Data config by machine translation.

A deterministic sample of collections (and of questions per collection) is translated once, and four configs with the
same rows in the same order are written to a local data repository:

- ``en_<name>``: the original English rows, the reference point;
- ``tr_<name>``: Turkish documents with Turkish questions, answers and choices;
- ``xlang_en2tr_<name>``: English documents with Turkish questions (a Turkish chat over an English memory);
- ``xlang_tr2en_<name>``: Turkish documents with English questions.

Each question is translated together with all its answer choices, so the reference answer stays exactly one of the
choices. Translations are cached in an append-only JSONL file keyed by model and source text: an interrupted run (for
example a Colab disconnect) resumes where it stopped, and a rerun only pays for what is missing. Embeddings are not
computed here; ``evaluate.py`` computes them on first use, and a rewritten config's stale embeddings are removed.

Run from the repository root: ``python -m tr_eval.build_data --source pv4 --collections 5 --questions_per_collection 80``.
"""

import argparse
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import random
import shutil
from typing import TYPE_CHECKING, Any

from datasets import Dataset
from tqdm import tqdm

from memorilla.data import (
    ANSWER_COLUMN,
    CHOICES_COLUMN,
    COLLECTION_COLUMN,
    DOCUMENTS_COLUMN,
    HARD_COLUMN,
    QUESTION_COLUMN,
    TEST_SPLIT,
    ensure_data,
    load_split,
)
from memorilla.metrics import choice_index
from memorilla.paths import DATA_DIR, DATA_REPO, resolve
from memorilla.utils import DEFAULT_SEED, write_json
from tr_eval.common import (
    USAGE,
    JsonlCache,
    complete,
    create_client,
    log_key_usage,
    parse_json_object,
    set_budget,
    stable_key,
)
from tr_eval.prompts import document_prompt, qa_prompt

if TYPE_CHECKING:
    from openai import OpenAI

LOGGER = logging.getLogger(__name__)

SOURCE_COLUMNS = [COLLECTION_COLUMN, QUESTION_COLUMN, ANSWER_COLUMN, DOCUMENTS_COLUMN, CHOICES_COLUMN, HARD_COLUMN]
ENGLISH = "en"
TURKISH = "tr"
VARIANTS = {  # config prefix -> (language of the documents, language of the question, answer and choices)
    "en": (ENGLISH, ENGLISH),
    "tr": (TURKISH, TURKISH),
    "xlang_en2tr": (ENGLISH, TURKISH),
    "xlang_tr2en": (TURKISH, ENGLISH),
}
DEFAULT_OUTPUT_REPO = "data_tr"
DEFAULT_COLLECTIONS = 10
DEFAULT_MODEL = "deepseek/deepseek-v4-flash"
DEFAULT_REASONING_EFFORT = "none"
DEFAULT_MAX_WORKERS = 32
DEFAULT_MAX_RETRIES = 3
CACHE_FILE = Path("translations") / "cache.jsonl"
MANIFEST_FILE = "tr_build.json"
EMBEDDINGS_DIR = "embeddings"


@dataclass(frozen=True)
class TranslatedQA:
    """A question with its answer choices in Turkish.

    Attributes:
        question: The translated question.
        answer: The translated reference answer (one of ``choices`` when the row has choices).
        choices: The translated choices, with empty choices kept empty, or None when the row has none.
    """

    question: str
    answer: str
    choices: list[str] | None


def sample_rows(collection_ids: list[int], num_collections: int, questions_per_collection: int, seed: int) -> list[int]:
    """Pick a deterministic sample of rows: whole collections first, then questions within each collection.

    Args:
        collection_ids: The collection of every row, in stored order.
        num_collections: Collections to keep; 0 keeps all of them.
        questions_per_collection: Rows to keep per collection; 0 keeps all of them.
        seed: Sampling seed.

    Returns:
        The selected row indices in stored order.
    """
    rows_by_collection: dict[int, list[int]] = defaultdict(list)
    for index, collection_id in enumerate(collection_ids):
        rows_by_collection[collection_id].append(index)
    rng = random.Random(seed)
    collections = sorted(rows_by_collection)
    if 0 < num_collections < len(collections):
        collections = rng.sample(collections, num_collections)
    selected = []
    for collection_id in sorted(collections):
        rows = rows_by_collection[collection_id]
        if 0 < questions_per_collection < len(rows):
            rows = rng.sample(rows, questions_per_collection)
        selected.extend(rows)
    return sorted(selected)


def qa_options(row: dict[str, Any]) -> tuple[list[str], list[int]]:
    """List the texts to translate with a row's question: its non-empty choices, or its answer when it has none.

    Args:
        row: A dataset row.

    Returns:
        The option texts and their positions in the row's ``choices`` (empty when the row has no choices).
    """
    choices = row.get(CHOICES_COLUMN)
    if not choices:
        return [str(row[ANSWER_COLUMN])], []
    positions = [index for index, choice in enumerate(choices) if str(choice).strip()]
    return [str(choices[index]) for index in positions], positions


def parse_qa(reply: str, sources: list[str]) -> tuple[str, list[str]]:
    """Parse and check a question-and-options translation.

    Args:
        reply: The translator's reply.
        sources: The English options, in order.

    Returns:
        The translated question and options, stripped.

    Raises:
        ValueError: If the reply is not the expected JSON object, the number of options differs, an option is
            empty, or options that differ in English coincide in Turkish.
    """
    value = parse_json_object(reply)
    question, options = value.get("question"), value.get("options")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("The translation has no question.")
    if not isinstance(options, list) or not all(isinstance(option, str) for option in options):
        raise ValueError("The translation's options are not a list of strings.")
    if len(options) != len(sources):
        raise ValueError(f"Expected {len(sources)} options, got {len(options)}.")
    options = [option.strip() for option in options]
    if not all(options):
        raise ValueError("The translation has an empty option.")
    if len({source.strip() for source in sources}) == len(sources) and len(set(options)) != len(options):
        raise ValueError("Distinct options were translated to the same text.")
    return question.strip(), options


class Translator:
    """Translates documents and questions with an LLM (OpenRouter or OpenAI), through a persistent cache."""

    def __init__(
        self, client: "OpenAI", model: str, reasoning_effort: str, max_retries: int, cache: JsonlCache
    ) -> None:
        """Set up the translator.

        Args:
            client: Client from ``tr_eval.common.create_client`` (or a test double; see ``tr_eval.common.complete``).
            model: Translation model.
            reasoning_effort: Reasoning effort for ``model``.
            max_retries: Attempts per request, and per unparsable question translation.
            cache: Cache of finished translations.
        """
        self.client = client
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_retries = max_retries
        self.cache = cache

    def _complete(self, prompt: str) -> str:
        """Send one prompt.

        Args:
            prompt: The prompt.

        Returns:
            The reply text.
        """
        return complete(self.client, prompt, self.model, self.reasoning_effort, self.max_retries)

    def document(self, text: str) -> str:
        """Translate one document.

        Args:
            text: The English document.

        Returns:
            The Turkish document (an empty document stays empty).
        """
        if not text.strip():
            return text
        key = stable_key("document", self.model, text)
        if key not in self.cache:
            self.cache.put(key, self._complete(document_prompt(text)).strip())
        return self.cache.get(key)

    def qa(self, question: str, options: list[str]) -> tuple[str, list[str]]:
        """Translate a question together with its options.

        Args:
            question: The English question.
            options: The English options.

        Returns:
            The Turkish question and options.

        Raises:
            RuntimeError: If no reply passes ``parse_qa`` within ``max_retries`` attempts.
        """
        key = stable_key("qa", self.model, question, options)
        if key not in self.cache:
            payload = json.dumps({"question": question, "options": options}, ensure_ascii=False, indent=2)
            prompt = qa_prompt(payload, len(options))
            for attempt in range(self.max_retries):
                try:
                    translated, translated_options = parse_qa(self._complete(prompt), options)
                    break
                except ValueError as error:
                    LOGGER.warning("Bad question translation (attempt %d/%d): %s", attempt + 1, self.max_retries, error)
            else:
                raise RuntimeError(f"No usable translation of the question {question!r}.")
            self.cache.put(key, {"question": translated, "options": translated_options})
        value = self.cache.get(key)
        return value["question"], value["options"]


def translate_row_qa(translator: Translator, row: dict[str, Any]) -> TranslatedQA:
    """Translate a row's question, answer and choices.

    Args:
        translator: The translator.
        row: A dataset row.

    Returns:
        The translation, with the answer at the reference's position among the choices.
    """
    options, positions = qa_options(row)
    question, translated = translator.qa(str(row[QUESTION_COLUMN]).strip(), options)
    if not positions:
        return TranslatedQA(question=question, answer=translated[0], choices=None)
    choices = [""] * len(row[CHOICES_COLUMN])
    for position, text in zip(positions, translated, strict=True):
        choices[position] = text
    answer = choices[choice_index(row[CHOICES_COLUMN], str(row[ANSWER_COLUMN]))]
    return TranslatedQA(question=question, answer=answer, choices=choices)


def translate_rows(
    translator: Translator, rows: list[dict[str, Any]], max_workers: int
) -> tuple[dict[str, str], dict[int, TranslatedQA], set[int]]:
    """Translate every distinct document and every row's question and choices, concurrently.

    Args:
        translator: The translator.
        rows: The sampled rows.
        max_workers: Concurrent requests.

    Returns:
        The document translations by source text, the question translations by row position, and the positions of
        the rows that could not be fully translated (a failed document fails every row of its collection).
    """
    documents = sorted({document for row in rows for document in row[DOCUMENTS_COLUMN]})
    translated_documents: dict[str, str] = {}
    translated_qa: dict[int, TranslatedQA] = {}
    failed_documents: set[str] = set()
    failed_rows: set[int] = set()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures: dict[Future, tuple[str, Any]] = {}
        for document in documents:
            futures[pool.submit(translator.document, document)] = ("document", document)
        for position, row in enumerate(rows):
            futures[pool.submit(translate_row_qa, translator, row)] = ("qa", position)
        for future in tqdm(as_completed(futures), total=len(futures), desc="translate"):
            kind, item = futures[future]
            try:
                result = future.result()
            except RuntimeError as error:
                LOGGER.error("Translation failed: %s", error)
                (failed_documents if kind == "document" else failed_rows).add(item)
                continue
            if kind == "document":
                translated_documents[item] = result
            else:
                translated_qa[item] = result
    for position, row in enumerate(rows):
        if failed_documents.intersection(row[DOCUMENTS_COLUMN]):
            failed_rows.add(position)
    return translated_documents, translated_qa, failed_rows


def build_variants(
    rows: list[dict[str, Any]],
    translated_documents: dict[str, str],
    translated_qa: dict[int, TranslatedQA],
    skip: set[int],
) -> dict[str, list[dict[str, Any]]]:
    """Assemble the rows of every variant from the originals and their translations.

    Args:
        rows: The sampled rows.
        translated_documents: Document translations by source text.
        translated_qa: Question translations by row position.
        skip: Positions of rows left out of every variant.

    Returns:
        The rows of each variant, keyed by config prefix, all with the same rows in the same order.
    """
    variants: dict[str, list[dict[str, Any]]] = {prefix: [] for prefix in VARIANTS}
    for position, row in enumerate(rows):
        if position in skip:
            continue
        qa = translated_qa[position]
        for prefix, (document_language, qa_language) in VARIANTS.items():
            documents = row[DOCUMENTS_COLUMN]
            if document_language == TURKISH:
                documents = [translated_documents.get(document, document) for document in documents]
            item = {COLLECTION_COLUMN: row[COLLECTION_COLUMN], DOCUMENTS_COLUMN: list(documents)}
            if qa_language == TURKISH:
                item |= {QUESTION_COLUMN: qa.question, ANSWER_COLUMN: qa.answer}
                if qa.choices is not None:
                    item[CHOICES_COLUMN] = qa.choices
            else:
                item |= {QUESTION_COLUMN: row[QUESTION_COLUMN], ANSWER_COLUMN: row[ANSWER_COLUMN]}
                if row.get(CHOICES_COLUMN):
                    item[CHOICES_COLUMN] = list(row[CHOICES_COLUMN])
            if HARD_COLUMN in row:
                item[HARD_COLUMN] = bool(row[HARD_COLUMN])
            variants[prefix].append(item)
    return variants


def read_manifest(root: Path) -> dict[str, Any]:
    """Read the build manifest of an output repository.

    Args:
        root: The output repository.

    Returns:
        The manifest (empty when there is none yet).
    """
    path = root / MANIFEST_FILE
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_config(root: Path, config: str, split: str, rows: list[dict[str, Any]], manifest: dict[str, Any]) -> bool:
    """Write one config's split as a single parquet shard, unless the manifest shows it is already up to date.

    A rewritten split loses its embeddings, which would otherwise no longer match its rows.

    Args:
        root: The output repository.
        config: Config name.
        split: Split name.
        rows: The rows.
        manifest: The build manifest; the split's digest is recorded in it.

    Returns:
        True when the split was (re)written.
    """
    digest = stable_key(rows)
    path = root / "data" / config / f"{split}-00000-of-00001.parquet"
    entry = manifest.setdefault(config, {})
    if path.exists() and entry.get(split) == digest:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    Dataset.from_list(rows).to_parquet(str(temporary))
    os.replace(temporary, path)
    for stale in (root / EMBEDDINGS_DIR).glob(f"*/{config}/{split}"):
        shutil.rmtree(stale)
    entry[split] = digest
    return True


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=str, required=True, help="Source config, e.g. pv4 or pmv2.")
    parser.add_argument("--split", type=str, default=TEST_SPLIT, help="Source split (also the output split).")
    parser.add_argument("--name", type=str, default=None, help="Suffix of the output configs (default: --source).")
    parser.add_argument(
        "--collections", type=int, default=DEFAULT_COLLECTIONS, help="Collections to sample; 0 keeps all."
    )
    parser.add_argument("--questions_per_collection", type=int, default=0, help="Rows per collection; 0 keeps all.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Sampling seed.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Translation model.")
    parser.add_argument("--reasoning_effort", type=str, default=DEFAULT_REASONING_EFFORT, help="Its reasoning effort.")
    parser.add_argument(
        "--max_cost",
        type=float,
        default=None,
        help="Stop sending OpenRouter requests once this run has spent this many USD (finished work stays cached).",
    )
    parser.add_argument("--max_workers", type=int, default=DEFAULT_MAX_WORKERS, help="Concurrent requests.")
    parser.add_argument("--max_retries", type=int, default=DEFAULT_MAX_RETRIES, help="Attempts per request.")
    parser.add_argument("--data_repo", type=str, default=DATA_REPO, help="Source data repo (Hub id or directory).")
    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR), help="Local mirror of a Hub data repo.")
    parser.add_argument(
        "--output_repo", type=str, default=DEFAULT_OUTPUT_REPO, help="Local data repo to write the configs to."
    )
    parser.add_argument("--cache", type=str, default=None, help=f"Translation cache (default <output>/{CACHE_FILE}).")
    parser.add_argument("--dry_run", action="store_true", help="Only report the sample size; no API calls.")
    parser.add_argument(
        "--allow_missing",
        action="store_true",
        help="Write the configs without the rows that could not be translated instead of stopping.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Sample, translate and write the four configs.

    Args:
        args: Parsed command-line arguments.

    Raises:
        SystemExit: If some rows could not be translated and ``--allow_missing`` is not set. Nothing is written then;
            the finished translations stay cached, so a rerun only retries the failed ones.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = ensure_data(args.source, [args.split], repo=args.data_repo, data_dir=args.data_dir)
    dataset = load_split(args.source, args.split, root, SOURCE_COLUMNS)
    indices = sample_rows(list(dataset[COLLECTION_COLUMN]), args.collections, args.questions_per_collection, args.seed)
    rows = dataset.select(indices).to_list()
    documents = {document for row in rows for document in row[DOCUMENTS_COLUMN]}
    LOGGER.info(
        "Sampled %d rows from %d collections: %d distinct documents (%d characters) and %d questions to translate.",
        len(rows),
        len({row[COLLECTION_COLUMN] for row in rows}),
        len(documents),
        sum(len(document) for document in documents),
        len(rows),
    )
    if args.dry_run:
        return

    output = resolve(args.output_repo)
    cache = JsonlCache(Path(args.cache) if args.cache else output / CACHE_FILE)
    translator = Translator(create_client(), args.model, args.reasoning_effort, args.max_retries, cache)
    set_budget(args.max_cost)
    log_key_usage()
    translated_documents, translated_qa, failed = translate_rows(translator, rows, args.max_workers)
    LOGGER.info("API usage (new requests only; cached translations are free): %s", USAGE.summary())
    log_key_usage()
    if failed and not args.allow_missing:
        raise SystemExit(
            f"{len(failed)} of {len(rows)} rows could not be translated, so nothing was written. Rerun to retry them "
            "(finished translations are cached), or pass --allow_missing to write the configs without them."
        )
    if failed:
        LOGGER.warning("Leaving out %d rows that could not be translated.", len(failed))

    name = args.name or args.source
    manifest = read_manifest(output)
    for prefix, variant_rows in build_variants(rows, translated_documents, translated_qa, failed).items():
        config = f"{prefix}_{name}"
        written = write_config(output, config, args.split, variant_rows, manifest)
        manifest[config]["source"] = {
            "config": args.source,
            "split": args.split,
            "collections": args.collections,
            "questions_per_collection": args.questions_per_collection,
            "seed": args.seed,
            "model": args.model,
        }
        LOGGER.info("%s %s/%s: %d rows", "Wrote" if written else "Unchanged", config, args.split, len(variant_rows))
    write_json(output / MANIFEST_FILE, manifest)


if __name__ == "__main__":
    main(parse_args())
