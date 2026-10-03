"""PEFT (LoRA) backend: text generation fine-tuning.

Implements the ``Backend`` protocol (CONTRACTS §2.2) for ``text_generation``,
model id ``lfm2-350m`` (``LiquidAI/LFM2-350M``, ungated, native
``transformers`` support as of 5.x -- no ``trust_remote_code``). LoRA SFT via
``peft``; merges the adapter before export so every exported format is a
plain dense checkpoint, not an adapter needing the base model alongside it.

Data convention (Track E1's data layer does not exist yet, same caveat as the
other two backends in this track):
  ``data.train`` / ``data.validation`` = ``list[{"prompt": str, "completion":
  str}]``. ``data.labels`` is unused (``None``).

Exports
-------
- ``safetensors``: the merged HF checkpoint (``model.safetensors`` + config +
  tokenizer), via ``save_pretrained(..., safe_serialization=True)``.
- ``gguf``: via llama.cpp's ``convert_hf_to_gguf.py`` on the merged
  safetensors dir, then ``llama-quantize`` to the format in
  ``SAKURA_GGUF_QUANT`` (default ``Q8_0``). Needs ``LLAMA_CPP_DIR`` pointing
  at a llama.cpp checkout with those two built (docs/atelier/envs/peft.txt).
  Run with ``sys.executable`` (this process' own interpreter), not a bare
  ``python3``: llama.cpp's convert script needs a ``transformers`` new
  enough to read whatever tokenizer backend class this repo's ``transformers``
  saved (found the hard way -- the host's system python3 had an older
  ``transformers`` that could not load LFM2's tokenizer; see the
  development notes). This venv additionally needs the ``gguf`` pip package
  (not a llama.cpp requirement installed by default outside its own repo).
- ``mlx``: via ``mlx_lm.convert``, only runs where ``mlx``/``mlx-lm`` import
  (Apple Silicon). On Linux this raises ``MLXUnavailableError`` with the
  precise reason (no Metal, pip has no linux-x86_64 wheel) -- never faked.

``runtime_eval``/``predict`` re-score the GGUF artifact through
``llama-cpp-python`` (``Llama(model_path, ...)(prompt, ...)``, its raw text
*completion* API, not ``create_chat_completion``) -- no chat template is
applied, so the model sees exactly the training-time prompt string, nothing
wrapped around it. An earlier version of this module shelled out to the
``llama-cli`` binary instead; on this build, llama-cli still
rendered its interactive REPL/chat framing around a "single-turn" completion
regardless of ``--no-conversation``/``--chat-template ''``, which measured
exact_match 0.0 for reasons that had nothing to do with the GGUF itself (measured
during development). ``llama-cpp-python`` needs a C/C++ toolchain to
build from source (``CC=gcc CXX=g++ pip install llama-cpp-python`` --
docs/atelier/envs/peft.txt); it is independent of the ``LLAMA_CPP_DIR``
checkout, which is still needed for ``export(..., "gguf")``'s conversion/
quantisation step. The safetensors/mlx formats are reported with
``value: None`` and a reason: GGUF is the only format this module has a
runtime re-scorer for in phase 1.

``runtime_eval`` protocol note: ``Backend.runtime_eval(self, artifact, data)``
(CONTRACTS §2.2) is not passed the ``TrainResult``, so ``matches_training`` is
left ``None`` here -- the engine's ``run`` loop (Track E1) holds
``TrainResult.metrics`` and is the right place to apply the §2.3 tolerance.

Licence: LFM2 weights are under the LFM Open License (CONTRACTS §2.4).
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sakura.atelier.types import (
    Artifact,
    Checkpoint,
    MaterialisedData,
    MetricsSink,
    ResolvedSpec,
    RunDir,
    RuntimeEval,
    TrainResult,
)

LICENSE = "LFM Open License v1.0 (LiquidAI/LFM2-350M)"

BASE_MODEL = "LiquidAI/LFM2-350M"


class MLXUnavailableError(RuntimeError):
    """mlx/mlx-lm does not run on this host -- raised with the precise cause,
    never silently skipped without saying why."""


class GGUFExportError(RuntimeError):
    """llama.cpp conversion/quantisation failed -- carries the subprocess's
    own stderr."""


@dataclass
class _Fitted:
    model_dir: Path  # merged HF checkpoint (safetensors) on disk
    tokenizer_dir: Path
    eval_examples: list[dict[str, str]]  # validation prompts+expected completions


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _llama_cpp_dir() -> Path:
    d = os.environ.get("LLAMA_CPP_DIR")
    if not d:
        raise GGUFExportError(
            "LLAMA_CPP_DIR is not set -- point it at a llama.cpp checkout with "
            "convert_hf_to_gguf.py, and build/bin/llama-quantize + llama-cli"
        )
    return Path(d)


def _exact_match(predicted: str, expected: str) -> bool:
    return predicted.strip().split("\n")[0].strip() == expected.strip()


def _load_llama_cpp(model_path: Path) -> Any:
    """Load a GGUF for raw-completion inference via llama-cpp-python -- no
    chat template, so the model sees exactly the training-time prompt
    string. Caller handles ImportError/load failures."""
    from llama_cpp import Llama

    return Llama(model_path=str(model_path), n_ctx=256, verbose=False)


class PEFTBackend:
    name = "peft"
    tasks = frozenset({"text_generation"})
    licence = LICENSE

    def train(
        self,
        spec: ResolvedSpec,
        data: MaterialisedData,
        out: RunDir,
        metrics: MetricsSink,
        resume: Checkpoint | None,
    ) -> TrainResult:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        out.ensure()
        hp = spec.hyperparameters
        epochs = int(hp.get("epochs", 3))
        lr = float(hp.get("learning_rate", 2e-4))
        batch_size = int(hp.get("batch_size", 8))
        lora_r = int(hp.get("lora_r", 8))
        lora_alpha = int(hp.get("lora_alpha", 16))
        max_len = int(hp.get("max_length", 64))

        device = _device()
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_dir = resume.path / "merged" if resume is not None else None
        model = AutoModelForCausalLM.from_pretrained(
            str(base_dir) if base_dir and base_dir.exists() else BASE_MODEL,
            dtype=torch.float32,
        ).to(device)

        lora_cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05,
            target_modules="all-linear", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        model.train()

        train_examples = list(data.train)
        val_examples = list(data.validation)

        def encode(ex: dict[str, str]) -> dict[str, torch.Tensor]:
            text = ex["prompt"] + ex["completion"] + tokenizer.eos_token
            enc = tokenizer(text, truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")
            prompt_len = len(tokenizer(ex["prompt"], truncation=True, max_length=max_len)["input_ids"])
            labels = enc["input_ids"].clone()
            labels[:, :prompt_len] = -100  # loss only on the completion
            labels[enc["attention_mask"] == 0] = -100
            return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}

        optim = torch.optim.AdamW(model.parameters(), lr=lr)
        rng = random.Random(spec.spec.seed)

        @torch.no_grad()
        def validate(step: int, epoch: float) -> tuple[float, float]:
            model.eval()
            losses = []
            correct = 0
            for ex in val_examples:
                enc = encode(ex)
                enc = {k: v.to(device) for k, v in enc.items()}
                out_ = model(**enc)
                losses.append(float(out_.loss))
                gen = model.generate(
                    input_ids=tokenizer(ex["prompt"], return_tensors="pt")["input_ids"].to(device),
                    max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.pad_token_id,
                )
                text = tokenizer.decode(gen[0], skip_special_tokens=True)
                completion = text[len(ex["prompt"]):]
                if _exact_match(completion, ex["completion"]):
                    correct += 1
            model.train()
            val_loss = sum(losses) / len(losses)
            exact_match = correct / len(val_examples)
            metrics.log("loss", val_loss, step=step, split="validation", epoch=epoch)
            metrics.log("exact_match", exact_match, step=step, split="validation", epoch=epoch)
            return val_loss, exact_match

        t0 = time.time()
        step = 0
        val_loss = val_em = 0.0
        for epoch in range(epochs):
            order = list(range(len(train_examples)))
            rng.shuffle(order)
            for i in range(0, len(order), batch_size):
                batch_idx = order[i:i + batch_size]
                batch = [encode(train_examples[j]) for j in batch_idx]
                input_ids = torch.cat([b["input_ids"] for b in batch]).to(device)
                attn = torch.cat([b["attention_mask"] for b in batch]).to(device)
                labels = torch.cat([b["labels"] for b in batch]).to(device)
                out_ = model(input_ids=input_ids, attention_mask=attn, labels=labels)
                out_.loss.backward()
                optim.step()
                optim.zero_grad()
                metrics.log("loss", float(out_.loss), step=step, split="train", epoch=epoch + i / len(order))
                step += 1
            val_loss, val_em = validate(step, float(epoch + 1))
        train_seconds = time.time() - t0

        merged = model.merge_and_unload()
        ckpt_dir = out.checkpoint
        merged_dir = ckpt_dir / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(str(merged_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(merged_dir))
        (ckpt_dir / "meta.json").write_text(json.dumps({"model": spec.spec.model, "task": "text_generation"}))

        fitted = _Fitted(model_dir=merged_dir, tokenizer_dir=merged_dir, eval_examples=val_examples)
        return TrainResult(
            model=fitted,
            metrics={"loss": val_loss, "exact_match": val_em},
            train_seconds=train_seconds,
            checkpoint=Checkpoint(path=ckpt_dir, model=spec.spec.model, task="text_generation"),
        )

    def export(self, result: TrainResult, fmt: str, out: RunDir) -> Artifact:
        out.ensure()
        fitted: _Fitted = result.model
        if fmt == "safetensors":
            dest = out.artifacts / "model.safetensors"
            shutil.copy(fitted.model_dir / "model.safetensors", dest)
            # Every OTHER file next to the merged weights (config.json,
            # tokenizer.json, tokenizer_config.json, special_tokens_map.json,
            # generation_config.json, ...) -- an earlier version of this
            # method copied only config.json, so AutoTokenizer.from_pretrained
            # on the exported artifact's own directory failed outright
            # (no tokenizer.json to find); the exported safetensors artifact
            # must be loadable standalone, same as the merged checkpoint is.
            for extra in fitted.model_dir.iterdir():
                if extra.is_file() and extra.name != "model.safetensors":
                    shutil.copy(extra, out.artifacts / extra.name)
            return _artifact(dest, "safetensors")
        if fmt == "gguf":
            return self._export_gguf(fitted, out)
        if fmt == "mlx":
            return self._export_mlx(fitted, out)
        raise ValueError(f"peft backend exports gguf|safetensors|mlx, got {fmt!r}")

    def _export_gguf(self, fitted: _Fitted, out: RunDir) -> Artifact:
        llama_dir = _llama_cpp_dir()
        convert_script = llama_dir / "convert_hf_to_gguf.py"
        quantize_bin = llama_dir / "build" / "bin" / "llama-quantize"
        if not convert_script.exists():
            raise GGUFExportError(f"{convert_script} not found")
        if not quantize_bin.exists():
            raise GGUFExportError(f"{quantize_bin} not found -- build llama.cpp first")

        fp16_path = out.artifacts / "model.fp16.gguf"
        proc = subprocess.run(
            [sys.executable, str(convert_script), str(fitted.model_dir), "--outfile", str(fp16_path), "--outtype", "f16"],
            capture_output=True, text=True, errors="replace",
        )
        if proc.returncode != 0:
            raise GGUFExportError(f"convert_hf_to_gguf.py failed: {proc.stderr[-4000:]}")

        quant = os.environ.get("SAKURA_GGUF_QUANT", "Q8_0")
        dest = out.artifacts / "model.gguf"
        proc = subprocess.run(
            [str(quantize_bin), str(fp16_path), str(dest), quant],
            capture_output=True, text=True, errors="replace",
        )
        if proc.returncode != 0:
            raise GGUFExportError(f"llama-quantize failed: {proc.stderr[-4000:]}")
        fp16_path.unlink(missing_ok=True)
        return _artifact(dest, "gguf")

    def _export_mlx(self, fitted: _Fitted, out: RunDir) -> Artifact:
        try:
            import mlx_lm  # noqa: F401
        except ImportError as exc:
            raise MLXUnavailableError(
                f"mlx_lm does not import on this host ({type(exc).__name__}: {exc}); "
                "mlx requires Apple Silicon (Metal) and ships no Linux x86_64 wheel"
            ) from exc
        dest_dir = out.artifacts / "model.mlx"
        proc = subprocess.run(
            [sys.executable, "-m", "mlx_lm.convert", "--hf-path", str(fitted.model_dir), "--mlx-path", str(dest_dir)],
            capture_output=True, text=True, errors="replace",
        )
        if proc.returncode != 0:
            raise MLXUnavailableError(f"mlx_lm.convert failed: {proc.stderr[-4000:]}")
        import hashlib

        total = b"".join(p.read_bytes() for p in sorted(dest_dir.rglob("*")) if p.is_file())
        return Artifact(path=dest_dir, format="mlx", sha256=hashlib.sha256(total).hexdigest(), bytes=len(total))

    def runtime_eval(self, artifact: Artifact, data: MaterialisedData) -> RuntimeEval:
        if artifact.format != "gguf":
            return RuntimeEval(
                format=artifact.format, runtime="none", metric="exact_match",
                value=None, n=0, matches_training=None,
                reason=f"this backend only re-scores gguf through llama-cpp-python; {artifact.format} has no runtime scorer in phase 1",
            )
        try:
            llm = _load_llama_cpp(artifact.path)
        except ImportError as exc:
            return RuntimeEval(
                format="gguf", runtime="llama-cpp-python", metric="exact_match",
                value=None, n=0, matches_training=None,
                reason=f"llama_cpp (llama-cpp-python) is not importable: {type(exc).__name__}: {exc}",
            )
        except Exception as exc:  # noqa: BLE001
            return RuntimeEval(
                format="gguf", runtime="llama-cpp-python", metric="exact_match",
                value=None, n=0, matches_training=None,
                reason=f"llama_cpp.Llama failed to load {artifact.path}: {type(exc).__name__}: {exc}",
            )

        examples = list(data.validation)
        correct = 0
        for ex in examples:
            # Raw completion, exactly the training prompt string: no chat
            # template, no system/role wrapping.
            out_ = llm(ex["prompt"], max_tokens=8, temperature=0.0, stop=["\n"])
            text = out_["choices"][0]["text"]
            if _exact_match(text, ex["completion"]):
                correct += 1
        exact_match = correct / len(examples) if examples else 0.0

        # matches_training is the engine's call (Track E1's `run` loop holds
        # TrainResult.metrics; this method's signature does not).
        return RuntimeEval(
            format="gguf", runtime="llama-cpp-python", metric="exact_match",
            value=exact_match, n=len(examples), matches_training=None, reason=None,
        )

    def predict(self, artifact: Artifact, inputs: Any) -> Any:
        prompt = inputs["prompt"] if isinstance(inputs, dict) else str(inputs)
        max_tokens = inputs.get("max_tokens", 64) if isinstance(inputs, dict) else 64
        if artifact.format == "gguf":
            llm = _load_llama_cpp(artifact.path)
            out_ = llm(prompt, max_tokens=max_tokens, temperature=0.0)
            return {"text": out_["choices"][0]["text"].strip()}
        if artifact.format in ("safetensors", "mlx"):
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_dir = artifact.path.parent if artifact.format == "safetensors" else artifact.path
            tok = AutoTokenizer.from_pretrained(str(model_dir))
            model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.float32)
            enc = tok(prompt, return_tensors="pt")
            gen = model.generate(**enc, max_new_tokens=max_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
            text = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            return {"text": text}
        raise ValueError(f"cannot predict from artifact format {artifact.format!r}")


def _artifact(path: Path, fmt: str) -> Artifact:
    import hashlib

    data = path.read_bytes()
    return Artifact(path=path, format=fmt, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


BACKEND = PEFTBackend()
