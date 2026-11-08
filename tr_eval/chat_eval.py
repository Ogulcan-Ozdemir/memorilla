"""Multi-turn chat evaluation: does memory still help when questions arrive one after another in one conversation?

The released checkpoints were trained and evaluated on single questions. Here the questions of one collection (one
user) are chained into sessions of ``--turns`` questions in a fixed, seeded order. At turn ``t`` the prompt holds the
earlier questions with the model's own earlier answers, then the new question; the memory tokens are recomputed for
the new question from its stored embedding, as in single-turn evaluation. Every session is also run without history,
which reproduces single-turn answers for exactly the same rows, so the effect of the conversation is a paired
difference per turn. Because the row order follows only the collection ids, the ``en_``, ``tr_`` and ``xlang_``
configs written by ``tr_eval.build_data`` get identical sessions.

Methods: ``memorilla`` (a checkpoint with ``--prompt``), and the text baselines ``closed_book`` and ``rag`` (top
``--top_k`` documents for the current question, placed in the current user turn only). When a prompt outgrows the
window, the oldest turns are dropped, never the system prompt or the current question.

Outputs, under ``<output_dir>`` for each of ``history`` and ``no_history``: ``turn-<t>/<config>/`` with
``predictions.jsonl`` and ``metrics.json`` (so ``tr_eval.score`` scores each turn like any other run), plus
``<config>/sessions.jsonl`` with every transcript and ``<config>/chat_summary.json`` with the accuracy per turn. Run from the
repository root, for example::

    python -m tr_eval.chat_eval --method memorilla --checkpoint checkpoints/pv4 --prompt en_prompt_reply_tr \\
        --config tr_pv4 --data_repo data_tr --output_dir results/tr/chat/pv4/en_prompt_reply_tr
"""

import argparse
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
import json
import os
from pathlib import Path
import random
from typing import TYPE_CHECKING, Any

from datasets import Dataset
import torch

from memorilla.benchmarks import CHOICE, HELPFUL_PROMPT, Benchmark, custom_benchmark
from memorilla.data import (
    ANSWER_COLUMN,
    COLLECTION_COLUMN,
    DEFAULT_SYSTEM_PROMPT,
    DOCUMENTS_COLUMN,
    EVAL_COLUMNS,
    QUESTION_COLUMN,
    TEST_SPLIT,
    apply_chat_template,
    ensure_data,
    memory_placeholder,
)
from memorilla.embeddings import EmbeddingStore, ensure_embeddings, load_split_with_embeddings
from memorilla.metrics import (
    SCORER_GPU_MEMORY_UTILIZATION,
    check_scoring_inputs,
    clean_generation,
    fit_choice_metrics,
    score_predictions,
)
from memorilla.paths import DATA_DIR, DATA_REPO, DEFAULT_DECODER, DEFAULT_ENCODER, MODEL_CACHE_DIR, resolve
from memorilla.utils import DEFAULT_SEED, release_gpu_memory, seed_everything, write_json, write_jsonl

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
    from vllm import LLM

    from memorilla.vllm import MemoryVLLM

MEMORILLA, CLOSED_BOOK, RAG = "memorilla", "closed_book", "rag"
METHODS = (MEMORILLA, CLOSED_BOOK, RAG)
HISTORY, NO_HISTORY = "history", "no_history"
DEFAULT_TURNS = 4
DEFAULT_TOP_K = 5
DEFAULT_MAX_LENGTH = 8192  # Memorilla prompts: system prompt, memory, history and question
BASELINE_MAX_LENGTH = 32_768  # the retrieval window of evaluate_baselines.py
DEFAULT_MAX_NEW_TOKENS = 384
GENERATION_RESERVE = 32
DEFAULT_BATCH_SIZE = 32
MEMORILLA_GPU_MEMORY_UTILIZATION = 0.6
BASELINE_GPU_MEMORY_UTILIZATION = 0.85
DOCUMENT_SEPARATOR = "\n\n"
SESSIONS_FILE = "sessions.jsonl"
CHAT_SUMMARY_FILE = "chat_summary.json"
VLLM_ENGINE_VARIABLE = "VLLM_USE_V1"
VLLM_MULTIPROC_VARIABLE = "VLLM_WORKER_MULTIPROC_METHOD"

Turn = tuple[str, str]  # (user message, assistant answer)


@dataclass(frozen=True)
class Session:
    """One conversation: questions of a single collection, asked in order.

    Attributes:
        index: Session number.
        collection_id: The collection (user) whose documents back every turn.
        rows: Dataset rows, one per turn.
    """

    index: int
    collection_id: int
    rows: tuple[int, ...]


@dataclass
class Request:
    """One turn to generate.

    Attributes:
        row: Dataset row of the question.
        history: Earlier turns of the session, oldest first (empty without history).
    """

    row: int
    history: list[Turn]


@dataclass
class TurnLog:
    """How many earlier turns had to be dropped to fit the window, per row."""

    dropped: dict[int, int] = field(default_factory=dict)


def build_sessions(collection_ids: Sequence[int], turns: int, seed: int) -> list[Session]:
    """Chain each collection's rows into sessions of ``turns`` questions in a seeded random order.

    Args:
        collection_ids: Collection id per dataset row.
        turns: Questions per session; a collection's last session may be shorter.
        seed: Shuffle seed.

    Returns:
        The sessions, by collection id and then by order.

    Raises:
        ValueError: If ``turns`` is not positive.
    """
    if turns < 1:
        raise ValueError(f"turns must be positive, got {turns}.")
    groups: dict[int, list[int]] = defaultdict(list)
    for row, collection_id in enumerate(collection_ids):
        groups[int(collection_id)].append(row)
    rng = random.Random(seed)
    sessions = []
    for collection_id in sorted(groups):
        rows = list(groups[collection_id])
        rng.shuffle(rows)
        for start in range(0, len(rows), turns):
            sessions.append(Session(len(sessions), collection_id, tuple(rows[start : start + turns])))
    return sessions


def chat_messages(system_prompt: str, history: Sequence[Turn], user: str) -> list[dict[str, str]]:
    """Render a conversation as chat messages.

    Args:
        system_prompt: System prompt (with the memory placeholder for Memorilla).
        history: Earlier turns, oldest first.
        user: The current user message.

    Returns:
        The messages.
    """
    messages = [{"role": "system", "content": system_prompt}]
    for question, answer in history:
        messages += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    messages.append({"role": "user", "content": user})
    return messages


def fit_history(
    history: Sequence[Turn], length: Callable[[Sequence[Turn]], int], budget: int
) -> tuple[list[Turn], int]:
    """Drop the oldest turns until the prompt fits the token budget.

    Args:
        history: Earlier turns, oldest first.
        length: Prompt length in tokens for a given history.
        budget: Prompt token budget.

    Returns:
        The kept turns and the number dropped.

    Raises:
        ValueError: If the prompt does not fit even without history.
    """
    kept = list(history)
    while length(kept) > budget:
        if not kept:
            raise ValueError(f"The prompt is longer than {budget} tokens even without history; raise --max_length.")
        kept = kept[1:]
    return kept, len(history) - len(kept)


def run_sessions(
    sessions: Sequence[Session],
    questions: Sequence[str],
    generate: Callable[[list[Request]], list[str]],
    with_history: bool,
) -> dict[int, str]:
    """Play every session turn by turn; the turn-``t`` requests of all sessions go out together.

    Args:
        sessions: The sessions.
        questions: User message per dataset row (question plus any suffix).
        generate: Generates one answer per request.
        with_history: Whether each turn sees the session's earlier turns.

    Returns:
        The cleaned answer per dataset row.
    """
    answers: dict[int, str] = {}
    for turn in range(max(len(session.rows) for session in sessions)):
        requests = [
            Request(
                session.rows[turn],
                [(questions[row], answers[row]) for row in session.rows[:turn]] if with_history else [],
            )
            for session in sessions
            if len(session.rows) > turn
        ]
        for request, text in zip(requests, generate(requests), strict=True):
            answers[request.row] = clean_generation(text)
    return answers


class MemorillaGenerator:
    """Generates turns with a memory checkpoint; the memory is conditioned on the current question."""

    def __init__(
        self,
        model: "MemoryVLLM",
        store: EmbeddingStore,
        dataset: Dataset,
        system_prompt: str,
        questions: Sequence[str],
        max_length: int,
        max_new_tokens: int,
        batch_size: int,
    ) -> None:
        """Configure the generator.

        Args:
            model: The memory-conditioned decoder.
            store: Embedding store of the split.
            dataset: The split (needs ``collection_id``).
            system_prompt: System prompt without the memory placeholder.
            questions: User message per row.
            max_length: Context length of the engine.
            max_new_tokens: Generation budget per turn.
            batch_size: Requests per generation call.
        """
        self.model = model
        self.store = store
        self.collections = list(dataset[COLLECTION_COLUMN])
        self.system_prompt = system_prompt + memory_placeholder(model.memory.num_memories)
        self.questions = questions
        self.budget = max_length - max_new_tokens - GENERATION_RESERVE
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.log = TurnLog()

    def render(self, history: Sequence[Turn], row: int) -> str:
        """Render one prompt.

        Args:
            history: Earlier turns.
            row: Row of the current question.

        Returns:
            The prompt text, ending with the assistant header.
        """
        messages = chat_messages(self.system_prompt, history, self.questions[row])
        return apply_chat_template(self.model.tokenizer, messages, add_generation_prompt=True)

    def __call__(self, requests: list[Request]) -> list[str]:
        """Generate one answer per request.

        Args:
            requests: The turns.

        Returns:
            The raw generations.
        """
        tokenizer = self.model.tokenizer
        prompts = []
        for request in requests:

            def length(history: Sequence[Turn], row: int = request.row) -> int:
                return len(tokenizer.encode(self.render(history, row), add_special_tokens=False))

            history, self.log.dropped[request.row] = fit_history(request.history, length, self.budget)
            prompts.append(self.render(history, request.row))
        texts = []
        for start in range(0, len(requests), self.batch_size):
            batch = requests[start : start + self.batch_size]
            encoded = tokenizer(
                prompts[start : start + self.batch_size], padding=True, padding_side="left", return_tensors="pt"
            )
            documents = [self.store.documents(self.collections[request.row]) for request in batch]
            width = max(item.shape[0] for item in documents)
            doc_embeds = torch.zeros(len(batch), width, documents[0].shape[-1], dtype=torch.bfloat16)
            doc_padding_mask = torch.ones(len(batch), width, dtype=torch.bool)
            for index, item in enumerate(documents):
                doc_embeds[index, : item.shape[0]] = item
                doc_padding_mask[index, : item.shape[0]] = False
            question_embeds = torch.stack([self.store.question(request.row) for request in batch])
            texts += self.model.generate(
                encoded["input_ids"],
                encoded["attention_mask"],
                doc_embeds=doc_embeds,
                doc_padding_mask=doc_padding_mask,
                question_embeds=question_embeds.to(torch.bfloat16),
                max_new_tokens=self.max_new_tokens,
            )
        return texts


class TextGenerator:
    """Generates turns with the plain decoder, optionally with retrieved documents in the current turn."""

    def __init__(
        self,
        engine: "LLM",
        tokenizer: "PreTrainedTokenizerBase",
        system_prompt: str,
        questions: Sequence[str],
        documents: Sequence[Sequence[str]] | None,
        max_length: int,
        max_new_tokens: int,
    ) -> None:
        """Configure the generator.

        Args:
            engine: Text-only vLLM engine.
            tokenizer: Decoder tokenizer.
            system_prompt: System prompt.
            questions: User message per row.
            documents: Retrieved documents per row, or None for closed book.
            max_length: Context length of the engine.
            max_new_tokens: Generation budget per turn.
        """
        self.engine = engine
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.questions = questions
        self.documents = documents
        self.budget = max_length - max_new_tokens - GENERATION_RESERVE
        self.max_new_tokens = max_new_tokens
        self.log = TurnLog()

    def encode(self, history: Sequence[Turn], row: int) -> list[int]:
        """Tokenise one prompt; retrieved documents go before the current question only.

        Args:
            history: Earlier turns.
            row: Row of the current question.

        Returns:
            The prompt token ids.
        """
        user = self.questions[row]
        if self.documents is not None:
            user = DOCUMENT_SEPARATOR.join(self.documents[row]) + DOCUMENT_SEPARATOR + user
        text = apply_chat_template(
            self.tokenizer, chat_messages(self.system_prompt, history, user), add_generation_prompt=True
        )
        return self.tokenizer.encode(text, add_special_tokens=False)

    def __call__(self, requests: list[Request]) -> list[str]:
        """Generate one answer per request.

        Args:
            requests: The turns.

        Returns:
            The raw generations.
        """
        from vllm import SamplingParams

        prompts = []
        for request in requests:

            def length(history: Sequence[Turn], row: int = request.row) -> int:
                return len(self.encode(history, row))

            history, self.log.dropped[request.row] = fit_history(request.history, length, self.budget)
            prompts.append({"prompt_token_ids": self.encode(history, request.row)})
        sampling = SamplingParams(max_tokens=self.max_new_tokens, temperature=0.0, top_p=1.0)
        outputs = self.engine.generate(prompts, sampling_params=sampling, use_tqdm=False)
        return [output.outputs[0].text for output in outputs]


def turn_jobs(
    benchmark: Benchmark,
    dataset: Dataset,
    sessions: Sequence[Session],
    answers: dict[int, str],
    output_dir: Path,
) -> list[tuple[Benchmark, Dataset, list[str], Path]]:
    """Split one mode's answers into a scoring job per turn.

    Args:
        benchmark: The target.
        dataset: The split.
        sessions: The sessions.
        answers: Answer per row.
        output_dir: The mode's directory; turn ``t`` goes to ``turn-<t>/<config>``.

    Returns:
        ``(benchmark, dataset, predictions, output_dir)`` per turn, for ``score_predictions``.
    """
    jobs = []
    for turn in range(max(len(session.rows) for session in sessions)):
        rows = [session.rows[turn] for session in sessions if len(session.rows) > turn]
        name = f"{output_dir.name}/turn-{turn + 1}/{benchmark.config}"
        jobs.append(
            (
                replace(benchmark, name=name),
                dataset.select(rows),
                [answers[row] for row in rows],
                output_dir / f"turn-{turn + 1}" / benchmark.config,
            )
        )
    return jobs


def write_sessions(
    path: Path, sessions: Sequence[Session], dataset: Dataset, answers: dict[int, str], log: TurnLog
) -> None:
    """Write every transcript.

    Args:
        path: Output file.
        sessions: The sessions.
        dataset: The split.
        answers: Answer per row.
        log: Dropped-turn counts per row.
    """
    questions = dataset[QUESTION_COLUMN]
    references = dataset[ANSWER_COLUMN]
    write_jsonl(
        path,
        [
            {
                "session": session.index,
                "collection_id": session.collection_id,
                "turns": [
                    {
                        "row": row,
                        "question": questions[row],
                        "answer": references[row],
                        "prediction": answers[row],
                        "dropped_turns": log.dropped.get(row, 0),
                    }
                    for row in session.rows
                ],
            }
            for session in sessions
        ],
    )


def summarize(summaries: dict[str, dict[str, Any]], jobs: Sequence[tuple[Benchmark, Dataset, list[str], Path]]) -> dict:
    """Collect one mode's per-turn metrics and their row-weighted mean.

    Args:
        summaries: ``metrics.json`` content keyed by job name, from ``score_predictions``.
        jobs: The mode's per-turn jobs.

    Returns:
        ``turns`` (metrics per turn) and ``overall`` (row-weighted means).
    """
    turns = [summaries[benchmark.name] for benchmark, _, _, _ in jobs]
    total = sum(turn["num_examples"] for turn in turns)
    names = [name for name in turns[0] if name != "num_examples"]
    overall = {name: round(sum(turn[name] * turn["num_examples"] for turn in turns) / total, 2) for name in names}
    return {"turns": turns, "overall": {**overall, "num_examples": total}}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Arguments to parse; None reads ``sys.argv``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Dataset config, e.g. tr_pv4.")
    parser.add_argument("--split", default=TEST_SPLIT, help="Split to evaluate.")
    parser.add_argument("--method", choices=METHODS, required=True, help="Memorilla or a text baseline.")
    parser.add_argument("--checkpoint", default=None, help="Memory checkpoint (for --method memorilla).")
    parser.add_argument("--system_prompt", default=None, help="System prompt (default: the method's usual one).")
    parser.add_argument("--turns", type=int, default=DEFAULT_TURNS, help="Questions per session.")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="Documents retrieved per turn for rag.")
    parser.add_argument("--max_length", type=int, default=None, help="Context length (default by method).")
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="Generation budget.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Memorilla requests per call.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=None, help="Decoder GPU memory fraction.")
    parser.add_argument(
        "--scorer_gpu_memory_utilization",
        type=float,
        default=SCORER_GPU_MEMORY_UTILIZATION,
        help="GPU memory fraction of the choice-scoring encoder.",
    )
    parser.add_argument("--decoder", default=DEFAULT_DECODER, help="Decoder.")
    parser.add_argument("--encoder", default=DEFAULT_ENCODER, help="Encoder of the stored embeddings and of scoring.")
    parser.add_argument("--data_repo", default=DATA_REPO, help="Hub dataset repo or local directory.")
    parser.add_argument("--data_dir", default=str(DATA_DIR), help="Local mirror of a Hub data repo.")
    parser.add_argument("--cache_dir", default=MODEL_CACHE_DIR, help="Model cache directory.")
    parser.add_argument("--output_dir", required=True, help="Run directory (history/ and no_history/ go below it).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Session order and generation seed.")
    args = parser.parse_args(argv)
    if (args.method == MEMORILLA) != (args.checkpoint is not None):
        parser.error("--checkpoint is required for --method memorilla and only applies to it.")
    return args


def main(args: argparse.Namespace) -> None:
    """Generate both modes of every session, then score each turn.

    Args:
        args: Parsed command-line arguments.
    """
    seed_everything(args.seed)
    root = ensure_data(args.config, [args.split], repo=args.data_repo, data_dir=args.data_dir)
    ensure_embeddings(
        args.config,
        [args.split],
        repo=args.data_repo,
        data_dir=args.data_dir,
        encoder=args.encoder,
        cache_dir=args.cache_dir,
    )
    columns = [*EVAL_COLUMNS, DOCUMENTS_COLUMN] if args.method == RAG else EVAL_COLUMNS
    dataset, store = load_split_with_embeddings(args.config, args.split, root, columns, args.encoder)
    benchmark = fit_choice_metrics(replace(custom_benchmark(args.config, CHOICE), split=args.split), dataset)
    check_scoring_inputs(benchmark, dataset)
    sessions = build_sessions(dataset[COLLECTION_COLUMN], args.turns, args.seed)
    questions = [str(question).strip() for question in dataset[QUESTION_COLUMN]]
    output_dir = resolve(args.output_dir)

    if args.method == MEMORILLA:
        from memorilla.memory import MemoryModule
        from memorilla.vllm import MemoryVLLM

        max_length = args.max_length or DEFAULT_MAX_LENGTH
        model = MemoryVLLM(
            args.decoder,
            MemoryModule.from_pretrained(resolve(args.checkpoint)),
            max_model_len=max_length,
            gpu_memory_utilization=args.gpu_memory_utilization or MEMORILLA_GPU_MEMORY_UTILIZATION,
            cache_dir=args.cache_dir,
        )
        make = partial(
            MemorillaGenerator,
            model,
            store,
            dataset,
            args.system_prompt or DEFAULT_SYSTEM_PROMPT,
            questions,
            max_length,
            args.max_new_tokens,
            args.batch_size,
        )
        engine_choice = None
    else:
        from transformers import AutoTokenizer

        from evaluate_baselines import retrieve, start_engine

        os.environ.setdefault(VLLM_MULTIPROC_VARIABLE, "spawn")
        engine_choice = os.environ.get(VLLM_ENGINE_VARIABLE)
        max_length = args.max_length or BASELINE_MAX_LENGTH
        documents = retrieve(dataset, store, args.top_k) if args.method == RAG else None
        tokenizer = AutoTokenizer.from_pretrained(args.decoder, cache_dir=args.cache_dir, trust_remote_code=True)
        engine_args = argparse.Namespace(
            decoder=args.decoder,
            cache_dir=args.cache_dir,
            gpu_memory_utilization=args.gpu_memory_utilization or BASELINE_GPU_MEMORY_UTILIZATION,
            max_num_seqs=None,
        )
        model = start_engine(engine_args, max_length, None)
        make = partial(
            TextGenerator,
            model,
            tokenizer,
            args.system_prompt or HELPFUL_PROMPT,
            questions,
            documents,
            max_length,
            args.max_new_tokens,
        )

    jobs: dict[str, list] = {}
    for mode in (HISTORY, NO_HISTORY):
        generator = make()
        answers = run_sessions(sessions, questions, generator, with_history=mode == HISTORY)
        mode_dir = output_dir / mode
        write_sessions(mode_dir / args.config / SESSIONS_FILE, sessions, dataset, answers, generator.log)
        jobs[mode] = turn_jobs(benchmark, dataset, sessions, answers, mode_dir)
    del model
    release_gpu_memory()
    if args.method != MEMORILLA and engine_choice is None:
        os.environ.pop(VLLM_ENGINE_VARIABLE, None)

    summaries = score_predictions(
        [job for mode_jobs in jobs.values() for job in mode_jobs],
        encoder=args.encoder,
        cache_dir=args.cache_dir,
        gpu_memory_utilization=args.scorer_gpu_memory_utilization,
    )
    for mode, mode_jobs in jobs.items():
        summary = summarize(summaries, mode_jobs)
        write_json(output_dir / mode / args.config / CHAT_SUMMARY_FILE, summary)
        print(f"{mode}: {json.dumps(summary['overall'])}")


if __name__ == "__main__":
    main(parse_args())
