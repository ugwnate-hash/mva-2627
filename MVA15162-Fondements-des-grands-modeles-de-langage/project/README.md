# The alien genome

Homework of the course "Foundations of Large Language Models" (Master MVA). The problem statement is
`alien-genome-homework.pdf`: read it first.

## Setup

With [uv](https://docs.astral.sh/uv/):

    uv sync           # creates .venv with the dependencies
    uv run pytest     # the public tests: they all fail until you write the code

Without uv (on Colab, for example): `pip install torch numpy safetensors huggingface_hub tokenizers matplotlib pytest`,
then run everything from the root of this folder.

## Layout

| Path | What it is |
|---|---|
| `alien/data.py` | Part 1: reading the genome |
| `alien/model.py` | Part 2: the transformer (the constructors are provided: keep the parameter names) |
| `alien/train.py` | Part 3: training functions, and your training script |
| `alien/analysis.py` | Parts 4 and 6: reading the models |
| `alien/checkpoint.py` | Provided: saving and loading checkpoints in the grader's format |
| `tests/` | The public tests, and `adapters.py`, which connects them to your code |
| `answers.json` | Your answers (template) |
| `make_submission.py` | Checks your submission and builds `submission.zip` |
| `AGENTS.md` | Instructions for coding agents: if you use one, start it from this folder (see the Rules of the statement) |

## Handing in

    uv run python make_submission.py

then rename `submission.zip` to `lastname_firstname.zip` and send it by email, as the statement explains. It refuses to build the zip if `alien/` uses a PyTorch module or
function that the rules of the statement forbid.
