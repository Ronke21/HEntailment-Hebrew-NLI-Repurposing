"""Checkpoint helpers shared by encoder_pipeline and decoder_pipeline."""
from __future__ import annotations
import logging, re, shutil
from pathlib import Path

_CKPT = re.compile(r"^checkpoint-(\d+)$")


def last_complete_checkpoint(out_ckpt, log: logging.Logger | None = None) -> str | None:
    """Newest checkpoint dir that has trainer_state.json, or None.

    transformers' get_last_checkpoint() returns the highest-numbered
    checkpoint-* dir whether or not it is complete. A SIGTERM (time limit,
    scancel) that lands mid-save leaves a dir with model weights but no
    trainer_state.json; Trainer.train(resume_from_checkpoint=...) then raises
    FileNotFoundError at once, every launch, until the unit burns its submit
    budget. Incomplete dirs are removed here so the next
    complete one (or a fresh start) is used instead.
    """
    root = Path(out_ckpt)
    if not root.is_dir():
        return None
    dirs = sorted((int(m.group(1)), p) for p in root.iterdir()
                  if p.is_dir() and (m := _CKPT.match(p.name)))
    for _, p in reversed(dirs):
        if (p / "trainer_state.json").is_file():
            return str(p)
        if log:
            log.warning("Removing incomplete checkpoint (no trainer_state.json): %s", p)
        shutil.rmtree(p, ignore_errors=True)
    return None
