"""Tests for the multi-turn chat evaluation (CPU only, with a fake decoder)."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from memorilla.benchmarks import CHOICE, custom_benchmark
from memorilla.data import MEMORY_TOKEN
from memorilla.embeddings import load_split_with_embeddings
from memorilla.utils import prepare_tokenizer
from tr_eval.chat_eval import (
    MemorillaGenerator,
    Request,
    build_sessions,
    fit_history,
    parse_args,
    run_sessions,
    summarize,
    turn_jobs,
)


def test_build_sessions_cover_each_collection_once_in_a_seeded_order() -> None:
    """Every row lands in exactly one session of its own collection, and the order depends only on the seed."""
    collections = [7, 3, 7, 7, 3, 7, 7]
    sessions = build_sessions(collections, turns=2, seed=1)
    assert sorted(row for session in sessions for row in session.rows) == list(range(len(collections)))
    assert all(collections[row] == session.collection_id for session in sessions for row in session.rows)
    assert [len(session.rows) for session in sessions] == [2, 2, 2, 1]  # collection 3, then 7 (five rows)
    assert [session.index for session in sessions] == [0, 1, 2, 3]
    assert build_sessions(collections, turns=2, seed=1) == sessions
    with pytest.raises(ValueError):
        build_sessions(collections, turns=0, seed=1)


def test_fit_history_drops_the_oldest_turns() -> None:
    """Turns are dropped from the front until the prompt fits; a prompt too long without history is an error."""
    history = [("q1", "a1"), ("q2", "a2"), ("q3", "a3"), ("q4", "a4")]

    def length(turns: list[tuple[str, str]]) -> int:
        return 10 + 5 * len(turns)

    assert fit_history(history, length, budget=20) == ([("q3", "a3"), ("q4", "a4")], 2)
    assert fit_history(history, length, budget=100) == (history, 0)
    with pytest.raises(ValueError):
        fit_history(history, length, budget=5)


def test_run_sessions_feeds_back_the_model_answers() -> None:
    """With history, turn t sees the session's earlier questions and the model's own answers; without, nothing."""
    sessions = build_sessions([1, 1, 1, 2], turns=3, seed=0)
    questions = ["q0", "q1", "q2", "q3"]
    seen: list[Request] = []

    def generate(requests: list[Request]) -> list[str]:
        seen.extend(requests)
        return [f" a{request.row} " for request in requests]

    answers = run_sessions(sessions, questions, generate, with_history=True)
    assert answers == {row: f"a{row}" for row in range(4)}
    first = sessions[0].rows
    last_turn = next(request for request in seen if request.row == first[2])
    assert last_turn.history == [(questions[first[0]], f"a{first[0]}"), (questions[first[1]], f"a{first[1]}")]

    seen.clear()
    run_sessions(sessions, questions, generate, with_history=False)
    assert all(request.history == [] for request in seen)


class FakeMemoryModel:
    """Stands in for ``MemoryVLLM``: records what ``generate`` receives."""

    def __init__(self, tokenizer: Any) -> None:
        """Keep the tokenizer and fake a two-token memory.

        Args:
            tokenizer: Tokenizer with the memory token registered.
        """
        self.tokenizer = tokenizer
        self.memory = SimpleNamespace(num_memories=2)
        self.calls: list[dict[str, torch.Tensor]] = []

    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs: Any) -> list[str]:
        """Record the inputs and answer with the batch position.

        Args:
            input_ids: Prompt ids.
            attention_mask: Attention mask.
            **kwargs: Memory inputs and the generation budget.

        Returns:
            One answer per row.
        """
        self.calls.append({"input_ids": input_ids, "attention_mask": attention_mask, **kwargs})
        return [f"answer {index}" for index in range(len(input_ids))]


def test_memorilla_generator_renders_history_and_memory(tokenizer: Any, data_repo: Path) -> None:
    """Prompts carry the memory placeholders and the history; memory inputs follow each request's row."""
    prepare_tokenizer(tokenizer)
    dataset, store = load_split_with_embeddings("toy", "train", data_repo)
    model = FakeMemoryModel(tokenizer)
    questions = [str(question).strip() for question in dataset["question"]]
    generator = MemorillaGenerator(model, store, dataset, "be helpful", questions, 512, 16, batch_size=2)

    prompt = generator.render([("who wrote hamlet", "shakespeare")], row=0)
    assert prompt.count(MEMORY_TOKEN) == 2
    assert prompt.index("shakespeare") < prompt.index(questions[0])

    texts = generator([Request(0, []), Request(1, [("q", "a")]), Request(2, [])])
    assert texts == ["answer 0", "answer 1", "answer 0"]
    first, second = model.calls
    assert first["doc_embeds"].shape[:2] == (2, 4)  # rows 0 and 1: two and four documents, padded to four
    assert first["doc_padding_mask"][0].tolist() == [False, False, True, True]
    assert torch.equal(first["question_embeds"][1], store.question(1).to(torch.bfloat16))
    assert second["question_embeds"].shape[0] == 1
    assert generator.log.dropped == {0: 0, 1: 0, 2: 0}

    tight = MemorillaGenerator(model, store, dataset, "be helpful", questions, 112, 16, batch_size=2)
    long_history = [("who wrote hamlet " * 3, "shakespeare " * 3)] * 4
    tight([Request(1, long_history)])
    assert 0 < tight.log.dropped[1] <= 4


def test_turn_jobs_and_summary(data_repo: Path) -> None:
    """Each turn becomes a scoring job of its own rows, and the summary weights turns by their rows."""
    dataset, _ = load_split_with_embeddings("toy", "train", data_repo)
    sessions = build_sessions(dataset["collection_id"], turns=2, seed=0)
    answers = {row: f"a{row}" for row in range(len(dataset))}
    benchmark = custom_benchmark("toy", CHOICE)
    jobs = turn_jobs(benchmark, dataset, sessions, answers, Path("out/history"))
    assert [job[0].name for job in jobs] == ["history/turn-1/toy", "history/turn-2/toy"]
    assert [len(job[1]) for job in jobs] == [2, 1]  # collections 3 and 7 start; only 7 has a second question
    assert jobs[1][3] == Path("out/history/turn-2/toy")
    assert jobs[1][2] == [answers[sessions[1].rows[1]]]

    summaries = {
        "history/turn-1/toy": {"accuracy": 50.0, "num_examples": 2},
        "history/turn-2/toy": {"accuracy": 80.0, "num_examples": 1},
    }
    summary = summarize(summaries, jobs)
    assert summary["overall"] == {"accuracy": 60.0, "num_examples": 3}
    assert [turn["accuracy"] for turn in summary["turns"]] == [50.0, 80.0]


def test_parse_args_requires_a_checkpoint_for_memorilla_only() -> None:
    """Memorilla needs a checkpoint, and the baselines refuse one."""
    common = ["--config", "tr_pv4", "--output_dir", "out"]
    assert parse_args([*common, "--method", "memorilla", "--checkpoint", "c"]).turns == 4
    assert parse_args([*common, "--method", "rag"]).checkpoint is None
    with pytest.raises(SystemExit):
        parse_args([*common, "--method", "memorilla"])
    with pytest.raises(SystemExit):
        parse_args([*common, "--method", "closed_book", "--checkpoint", "c"])
