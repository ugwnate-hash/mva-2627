"""Check your submission and build submission.zip. Provided: run it before handing in.

Usage, from the root of the project:
    python make_submission.py

It checks that answers.json is complete, that your package uses none of the forbidden PyTorch modules and functions
(if it does, no zip is written), that it imports the way the grader imports it (offline, with only the dependencies of
the handout), that the three checkpoints load and respect the rules, measures them on the training
and validation sets (with the synonyms of your answers.json), runs the public tests, and writes submission.zip with
alien/, tests/adapters.py, answers.json and the three checkpoints. Rename that zip lastname_firstname.zip and send
it by email, as the problem statement explains.
The grader runs other tests and uses a private test set: passing these checks does not guarantee the points.
"""

import ast
import json
import os
import subprocess
import sys
import tempfile
import zipfile

import torch

from alien.checkpoint import load_checkpoint
from alien.data import build_vocabulary, encode, load_split
from alien.model import count_parameters

CHECKPOINTS = ("generic", "focused_memorized", "focused_final")
FOCUSED = dict(d_model=16, n_layers=1, n_heads=4, d_mlp=512, attn_window=2, rope_base=10000.0)
ALLOWED = {"torch", "numpy", "safetensors", "huggingface_hub", "tokenizers", "matplotlib", "pytest", "alien", "tests"}
MAX_MB = 20
# The PyTorch modules and functions that the rules of the homework forbid in alien/.
FORBIDDEN = {"scaled_dot_product_attention", "MultiheadAttention", "RMSNorm", "rms_norm", "cross_entropy",
             "log_softmax", "Transformer", "TransformerEncoder", "TransformerDecoder", "TransformerEncoderLayer",
             "TransformerDecoderLayer"}
problems = []


def problem(message):
    problems.append(message)
    print("  PROBLEM:", message)


def check_answers():
    print("answers.json")
    try:
        answers = json.load(open("answers.json"))
    except Exception as e:
        problem(f"cannot read answers.json: {e}")
        return None
    for key in ("codon_length", "p"):
        if not isinstance(answers.get(key), int):
            problem(f"answers.json: '{key}' must be an integer")
    if not answers.get("synonyms"):
        problem("answers.json: 'synonyms' is empty")
    rule = answers.get("rule") or {}
    if not all(isinstance(rule.get(k), int) for k in ("a", "b", "c")):
        problem("answers.json: 'rule' needs integers a, b and c")
    window = (answers.get("generic") or {}).get("accuracy_by_window") or {}
    if set(window) != {"1", "2", "3", "full"} or not all(isinstance(v, (int, float)) for v in window.values()):
        problem("answers.json: generic.accuracy_by_window needs numbers for the keys '1', '2', '3' and 'full'")
    if len((answers.get("focused") or {}).get("key_frequencies") or []) != 3:
        problem("answers.json: focused.key_frequencies needs 3 frequencies")
    labels = answers.get("labels") or {}
    if not labels or not all(isinstance(v, int) for v in labels.values()):
        problem("answers.json: 'labels' must give an integer label to every codon")
    return answers


def find_forbidden(path):
    """The forbidden PyTorch names that the file at path uses, as (line, name) pairs. A name counts when it is taken
    from torch (torch.nn.functional.cross_entropy, F.cross_entropy, nn.RMSNorm, from torch.nn import RMSNorm...), and
    log_softmax counts on any object, since tensors have it as a method. Your own functions and classes with the same
    names (cross_entropy, RMSNorm) are allowed."""
    tree = ast.parse(open(path).read())
    from_torch, found = set(), []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "torch":
                    from_torch.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and (node.module or "").split(".")[0] == "torch":
            for alias in node.names:
                if alias.name in FORBIDDEN or alias.name == "*":
                    found.append((node.lineno, f"{node.module}.{alias.name}"))
                else:
                    from_torch.add(alias.asname or alias.name)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN:
            root = node.value
            while isinstance(root, ast.Attribute):
                root = root.value
            if node.attr == "log_softmax" or (isinstance(root, ast.Name) and root.id in from_torch):
                found.append((node.lineno, node.attr))
    return sorted(set(found))


def check_forbidden():
    """True if no file of alien/ uses a forbidden PyTorch module or function."""
    print("forbidden PyTorch modules and functions")
    ok = True
    for path in sorted(os.path.join("alien", f) for f in os.listdir("alien") if f.endswith(".py")):
        for line, name in find_forbidden(path):
            problem(f"{path}, line {line}: uses {name}, which the rules forbid")
            ok = False
    return ok


def check_imports():
    """The grader imports tests/adapters.py offline, with an empty Hugging Face cache, and with only the packages of
    the handout: data must not be loaded at import time, and no other package may be imported."""
    print("imports")
    files = [os.path.join("alien", f) for f in os.listdir("alien") if f.endswith(".py")] + ["tests/adapters.py"]
    for path in files:
        for node in ast.walk(ast.parse(open(path).read())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module] if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module else []
            for name in names:
                top = name.split(".")[0]
                if top not in ALLOWED and top not in sys.stdlib_module_names:
                    problem(f"{path} imports '{top}', which the grader does not have")
    with tempfile.TemporaryDirectory() as empty:
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_HUB_CACHE": empty, "HF_HOME": empty}
        result = subprocess.run([sys.executable, "-c", "import tests.adapters"], env=env, capture_output=True, text=True)
    if result.returncode != 0:
        last = (result.stderr.strip().splitlines() or ["?"])[-1]
        problem(f"importing tests/adapters.py offline fails ({last}): does a file of alien/ load data or train "
                "outside `if __name__ == \"__main__\":`?")


def accuracy(model, data, amino_acid_of):
    with torch.no_grad():
        logits = model(data[:, :-1])[:, 1:].reshape(-1, len(amino_acid_of))
    targets = data[:, 2:].reshape(-1)
    p_aa = torch.zeros(len(logits), int(amino_acid_of.max()) + 1).index_add_(1, amino_acid_of, logits.softmax(-1))
    return (p_aa.argmax(-1) == amino_acid_of[targets]).float().mean().item()


def check_checkpoints(answers):
    codon_length = (answers or {}).get("codon_length")
    if not isinstance(codon_length, int):
        problem("answers.json: codon_length is needed to read the corpus; checkpoints not checked")
        return
    train_lines = load_split("train")
    vocabulary = build_vocabulary(train_lines, codon_length)
    train = encode(train_lines[:2000], vocabulary, codon_length)
    valid = encode(load_split("valid"), vocabulary, codon_length)
    amino_acid_of = None
    group = {c: i for i, pair in enumerate((answers or {}).get("synonyms") or []) for c in pair}
    if all(c in group for c in vocabulary):
        amino_acid_of = torch.tensor([group[c] for c in vocabulary])
    else:
        print("  (the synonyms of answers.json do not cover every codon: accuracies on amino acids not measured)")
    for name in CHECKPOINTS:
        print(f"checkpoints/{name}")
        try:
            model, config = load_checkpoint(os.path.join("checkpoints", name))
        except Exception as e:
            problem(f"{name}: cannot load: {type(e).__name__}: {e}")
            continue
        n = count_parameters(model)
        print(f"  {n:,} parameters, step {config.get('step')}, attention window {config.get('attn_window')}")
        if n > 1_000_000:
            problem(f"{name}: more than 1,000,000 parameters")
        if config.get("vocabulary") != vocabulary or config.get("vocab_size") != len(vocabulary):
            problem(f"{name}: the vocabulary must be all the codons, in alphabetical order")
        if name.startswith("focused"):
            for key, value in FOCUSED.items():
                if config.get(key) != value:
                    problem(f"{name}: {key} must be {value}")
        elif config.get("attn_window") != 0:
            problem("generic: attn_window must be 0 (full attention)")
        if amino_acid_of is not None:
            print(f"  accuracy on amino acids: training {accuracy(model, train, amino_acid_of):.4f}, "
                  f"validation {accuracy(model, valid, amino_acid_of):.4f}")


def run_tests():
    print("public tests")
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], capture_output=True, text=True)
    print("  " + (result.stdout.strip().splitlines() or ["no output"])[-1])


def make_zip():
    files = ["answers.json", "tests/adapters.py"]
    for root, dirs, names in os.walk("alien"):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        files += [os.path.join(root, n) for n in names if n.endswith(".py")]
    files += [os.path.join("checkpoints", c, f) for c in CHECKPOINTS for f in ("model.safetensors", "config.json")]
    with zipfile.ZipFile("submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for path in files:
            if os.path.exists(path):
                z.write(path)
    size = os.path.getsize("submission.zip") / 1e6
    print(f"wrote submission.zip ({size:.1f} MB)")
    if size > MAX_MB:
        problem(f"submission.zip is larger than {MAX_MB} MB")


if __name__ == "__main__":
    answers = check_answers()
    allowed = check_forbidden()
    check_imports()
    check_checkpoints(answers)
    run_tests()
    if allowed:
        make_zip()
    else:
        print("\nNo submission.zip written: replace the forbidden modules and functions with your own code first.")
    print(f"\n{len(problems)} problem(s) found." if problems else "\nNo problem found.")
