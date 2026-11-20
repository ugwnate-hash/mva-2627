"""Saving and loading checkpoints in the format of the grader. Provided: do not change this file.

A checkpoint is a folder with two files:
    model.safetensors   the weights (the state_dict of TransformerLM, in float32)
    config.json         the fields of ModelConfig, the training step, and the vocabulary (codons in token id order)
"""

import dataclasses
import json
import os

import torch
from safetensors.torch import load_file, save_file

from .model import ModelConfig, TransformerLM


def save_checkpoint(model: TransformerLM, folder: str, step: int, vocabulary: list[str]):
    os.makedirs(folder, exist_ok=True)
    weights = {name: t.detach().to("cpu", torch.float32).contiguous() for name, t in model.state_dict().items()}
    save_file(weights, os.path.join(folder, "model.safetensors"))
    config = {**dataclasses.asdict(model.cfg), "step": int(step), "vocabulary": list(vocabulary)}
    with open(os.path.join(folder, "config.json"), "w") as f:
        json.dump(config, f, indent=1)


def load_checkpoint(folder: str) -> tuple[TransformerLM, dict]:
    """The model (in evaluation mode, on the CPU) and the content of config.json."""
    with open(os.path.join(folder, "config.json")) as f:
        config = json.load(f)
    fields = {f.name for f in dataclasses.fields(ModelConfig)}
    model = TransformerLM(ModelConfig(**{k: v for k, v in config.items() if k in fields}))
    model.load_state_dict(load_file(os.path.join(folder, "model.safetensors")))
    return model.eval(), config
