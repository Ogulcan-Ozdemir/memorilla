# Turkish evaluation

Tools to test whether the released Memorilla checkpoints (`memorilla/Memorilla-Qwen3-8B`, trained on English data only)
work when the user chats in Turkish. Everything runs from the repository root and needs the `tr` extra
(`pip install -e ".[tr]"`); the translation and the judge also need an API key (see [Models](#models)).

| Module | What it does | Needs |
| --- | --- | --- |
| `python -m tr_eval.build_data` | Samples collections from a test split, machine-translates them once and writes four configs with the same rows: `en_<src>`, `tr_<src>`, `xlang_en2tr_<src>` (English memory, Turkish question) and `xlang_tr2en_<src>`. | CPU, LLM API |
| `python -m tr_eval.run_queue` | `plan` writes the job table, `run` works through it (resumable, with an optional copy to Google Drive after every job), `status` shows progress. | GPU for `run` |
| `python -m tr_eval.score` | Adds Turkish-aware scores to every run: language fidelity, chrF++, optional embedding similarity and an LLM judge, then writes `summary.csv`/`summary.md` and an optional blind sheet for human raters. | CPU (GPU for `--embedding_similarity`), LLM API for the judge |

Choice accuracy, computed by `evaluate.py` itself, is the primary automatic metric: it compares embeddings and works in
any language. The repository's BLEU, ROUGE and METEOR are not reliable for Turkish (see the docstring of
`tr_eval/score.py`).

## Models

With `OPENROUTER_API_KEY` set, both scripts call [OpenRouter](https://openrouter.ai) (Chat Completions with its
`reasoning` parameter), log tokens, the cost OpenRouter reports and the key's total spend, and stop sending requests once
a run has spent `--max_cost` USD (finished work stays cached, so a rerun resumes). Otherwise they call the OpenAI
Responses API with `OPENAI_API_KEY`; then pass OpenAI model names (for example `--model gpt-5.1`).

The defaults are chosen for a total budget of about $1 (prices per million input/output tokens, November 2026; scores
from the Artificial Analysis Intelligence Index):

| Role | Default | Effort | Why |
| --- | --- | --- | --- |
| Translation (`--model`, `--reasoning_effort`) | `deepseek/deepseek-v4-flash` ($0.091/$0.182) | `none` | Translation is output-bound and needs no reasoning. In a pilot on two PV4 documents it kept the speaker labels and the Turkish technical terms ("ITAB" for HAZ) and the user's casual register, at the lowest cost; `xiaomi/mimo-v2.6-flash` (index 27) translated the `Assistant:` label and coined a term, and `anthropic/claude-haiku-5.5` cost 2.5 times as much and made the register formal. MiniMax-M3.1-Flash and GLM-5.3-Flash force reasoning. |
| Judge (`--judge_model`, `--judge_reasoning_effort`) | `anthropic/claude-haiku-5.5` ($0.10/$0.50) | `medium` | Judge prompts are input-heavy with short replies, so the low input price matters; it scores 25-36 depending on effort and comes from a different family than the translator and the evaluated model. |

Also set a credit limit on the OpenRouter key itself (Settings → API Keys); that is the hard cap.

## Sizes

PV4 collections hold about 140 questions over about 210K characters of documents, PMV2 collections about 25 questions
over about 160K characters, so the cost is driven by the number of collections. `--collections 5
--questions_per_collection 80` gives 400 PV4 rows (a 95% interval of about ±5 points on one accuracy) and about 125
PMV2 rows (about ±9 points). The English and Turkish configs share their rows, so differences between them are paired
and tighter. Run `build_data` with `--dry_run` first; it reports the number of documents and characters to translate.

## Running on Google Colab

Working files live on the runtime's local disk (`/content/work`) and are copied to Google Drive (`$DRIVE`) for
safekeeping. Paths under `$DRIVE` mirror those under `$MEMORILLA_HOME`, so one `rsync` restores a session.

**Setup**, after every connect (Python cell). Add an `OPENROUTER_API_KEY` (or `OPENAI_API_KEY`) secret in Colab's
Secrets panel, with notebook access, first if the session translates or judges:

```python
from google.colab import drive, userdata
import os

drive.mount("/content/drive")
os.environ.update(
    MEMORILLA_HOME="/content/work",
    MEMORILLA_MODEL_CACHE="/content/hf_cache",
    HF_HOME="/content/hf_cache",
    DRIVE="/content/drive/MyDrive/memorilla_tr",
    PY="/content/venv/bin/python",
)
for key in ("OPENROUTER_API_KEY", "OPENAI_API_KEY"):
    try:
        os.environ[key] = userdata.get(key)
        print(f"{key} loaded")
    except Exception:
        pass
```

```bash
%%bash
set -e
nvidia-smi --query-gpu=name,memory.total --format=csv || echo "CPU runtime"
[ -d /content/memorilla ] || git clone -b tr-eval https://github.com/Ogulcan-Ozdemir/memorilla /content/memorilla
cd /content/memorilla && git pull --ff-only
pip -q install uv
[ -x "$PY" ] || uv venv /content/venv --python 3.11
uv pip install --python "$PY" -q -e ".[tr]"
mkdir -p "$MEMORILLA_HOME" "$DRIVE" && rsync -a "$DRIVE/" "$MEMORILLA_HOME/"
```

The repository's pinned stack (torch 2.7.0, vLLM 0.9.1) goes into its own virtual environment, so Colab's preinstalled
packages stay untouched; every script runs with `$PY`.

**Data** (CPU runtime). About 1.9M characters for both sources; expected to cost about $0.35, and the caps stop it at $0.60:

```bash
%%bash
cd /content/memorilla
for src in pv4 pmv2; do
  $PY -m tr_eval.build_data --source $src --collections 5 --questions_per_collection 80 --max_cost 0.30
done
rsync -a "$MEMORILLA_HOME/data_tr" "$DRIVE/"
```

**Evaluation** (GPU runtime, A100 40 GB or larger; T4 has no bfloat16 and does not work). On a Blackwell GPU such as
the RTX PRO 6000 (sm_120), the pinned torch 2.7.0+cu126 and the vLLM 0.9.1 wheel have no kernels for the GPU; run
`bash tr_eval/setup_blackwell.sh` once per session (it builds vLLM for sm_120 the first time, about 16 minutes, and
keeps the wheel in `$DRIVE/wheels`), then use `PY=/content/venv128/bin/python` in the cells below. Download the checkpoints,
plan the jobs and start the queue in the background:

```bash
%%bash
cd /content/memorilla
/content/venv/bin/hf download memorilla/Memorilla-Qwen3-8B --include "pv4/*" "pmv2/*" "stage3/*" \
    --local-dir "$MEMORILLA_HOME/checkpoints"
mkdir -p "$MEMORILLA_HOME/results/tr"
[ -f "$MEMORILLA_HOME/results/tr/runs.tsv" ] || \
    $PY -m tr_eval.run_queue plan --tiers core extended > "$MEMORILLA_HOME/results/tr/runs.tsv"
nohup $PY -m tr_eval.run_queue run "$MEMORILLA_HOME/results/tr/runs.tsv" --sync_to "$DRIVE" \
    >> "$MEMORILLA_HOME/queue.log" 2>&1 &
```

Check on it as often as needed:

```bash
%%bash
cd /content/memorilla
$PY -m tr_eval.run_queue status "$MEMORILLA_HOME/results/tr/runs.tsv"
tail -n 3 "$MEMORILLA_HOME/queue.log"
```

After a disconnect, rerun the setup cells and the queue cell: finished jobs are skipped. On an L4 (24 GB), plan with
`--eval_args "--gpu_memory_utilization 0.85"` and leave out the `full` tier. Add `full` (the full-context baseline,
65K tokens with YaRN for PV4) only on a 40 GB or larger GPU.

**Scoring**, once the queue is done:

```bash
%%bash
cd /content/memorilla
$PY -m tr_eval.score --judge_samples 40 --judge_include "*/tr_*" "*/xlang_en2tr_*" --max_cost 0.30 \
    --human_sheet 50   # add --embedding_similarity on a GPU runtime
rsync -a "$MEMORILLA_HOME/results" "$DRIVE/"
```

The judge rates the 14 runs with Turkish questions, 40 rows each (about $0.20); try `--judge_samples 5` first to see
the cost per judgement. `results/tr/summary.md` holds one row per run. `results/tr/human_ratings.csv` is the blind sheet for native speakers;
`human_key.csv` maps its items back to runs and rows.
