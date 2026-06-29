"""Smoke tests for training.train — config plumbing, checkpoint/resume,
end-of-epoch artifacts. Uses CPU + tiny synthetic Dataset to keep fast."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from training.train import RunConfig, train


@pytest.fixture()
def tiny_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Build a 8-patch synthetic manifest + bundle + a stubbed catalog
    (unused, but bundle path skips it anyway).
    """
    n_train, n_val = 6, 4
    n = n_train + n_val
    rng = np.random.default_rng(0)
    bundle = rng.integers(0, 256, size=(n, 256, 256), dtype=np.uint8)
    bundle_path = tmp_path / "patches.npy"
    np.save(bundle_path, bundle)

    rows = []
    for i in range(n):
        rows.append({
            "product_id": f"P{i:04d}",
            "split": "train" if i < n_train else "val",
            "row0": 0, "col0": 0,
            "source_bits": "msb" if i % 2 == 0 else "lsb",
            "source_tdi": 64 if i % 2 == 0 else 128,
            "mean_dn": 50.0, "frac_zero": 0.0, "frac_sat": 0.0, "sobel_99": 30.0,
        })
    import pandas as pd
    df = pd.DataFrame(rows)
    manifest_path = tmp_path / "manifest.parquet"
    df.to_parquet(manifest_path)
    return manifest_path, bundle_path


def _cfg(run_dir: Path, manifest: Path, bundle: Path, *, epochs: int) -> RunConfig:
    return RunConfig(
        bundle=str(bundle),
        manifest=str(manifest),
        loss="mse",
        epochs=epochs,
        batch_size=2,
        lr=1e-4,
        weight_decay=0.0,
        warmup_steps=2,
        val_every_steps=3,
        ckpt_every_steps=2,
        val_batches=2,
        num_workers=0,
        device="cpu",
        seed=0,
        val_seed=99,
        run_dir=str(run_dir),
        drive_mirror=None,
        dark_threshold_dn=20.0,
        n_sample_pngs=1,
    )


def test_fresh_run_emits_artifacts(tiny_corpus, tmp_path):
    manifest, bundle = tiny_corpus
    run_dir = tmp_path / "run_fresh"
    train(_cfg(run_dir, manifest, bundle, epochs=1))

    assert (run_dir / "config.json").exists()
    assert (run_dir / "train_log.csv").exists()
    assert (run_dir / "latest.pt").exists()
    # At least one validation cycle happened (val_every=3 ≤ steps).
    assert (run_dir / "val_log.jsonl").exists()
    assert (run_dir / "best.pt").exists()
    # Sample PNGs.
    assert any((run_dir / "samples").glob("step*.png"))

    # config.json embeds git_sha and the total step count.
    cfg = json.loads((run_dir / "config.json").read_text())
    assert "git_sha" in cfg
    assert cfg["total_steps"] > 0
    assert cfg["bundle"] == str(bundle)

    # train_log header + at least one row.
    with (run_dir / "train_log.csv").open() as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["step", "epoch", "lr", "loss", "wall_s"]
    assert len(rows) >= 2


def test_resume_continues_from_latest(tiny_corpus, tmp_path):
    """Second call with the same run_dir resumes at the recorded step."""
    manifest, bundle = tiny_corpus
    run_dir = tmp_path / "run_resume"

    # Phase 1: train 1 epoch.
    train(_cfg(run_dir, manifest, bundle, epochs=1))
    state1 = torch.load(run_dir / "latest.pt", map_location="cpu", weights_only=False)
    step_after_first = int(state1["step"])
    assert step_after_first > 0

    # Phase 2: train 2 epochs (same run-dir, should resume + train 1 more).
    train(_cfg(run_dir, manifest, bundle, epochs=2))
    state2 = torch.load(run_dir / "latest.pt", map_location="cpu", weights_only=False)
    assert int(state2["step"]) > step_after_first
    assert int(state2["epoch"]) == 2


def test_pg_nll_loss_path_runs(tiny_corpus, tmp_path):
    """The pg_nll path threads sigma_fpn + g and produces a finite loss curve."""
    manifest, bundle = tiny_corpus
    run_dir = tmp_path / "run_pgnll"
    cfg = _cfg(run_dir, manifest, bundle, epochs=1)
    cfg.loss = "pg_nll"
    train(cfg)

    with (run_dir / "train_log.csv").open() as f:
        rows = list(csv.reader(f))
    losses = [float(r[3]) for r in rows[1:]]
    assert all(np.isfinite(losses))


def test_config_has_started_at_and_best_meta_written(tiny_corpus, tmp_path):
    """config.json embeds run start time; best.meta.json sidecar tracks
    the best.pt save (step/epoch/val_loss/psnr_dark)."""
    manifest, bundle = tiny_corpus
    run_dir = tmp_path / "run_meta"
    train(_cfg(run_dir, manifest, bundle, epochs=1))

    cfg = json.loads((run_dir / "config.json").read_text())
    assert "started_at" in cfg
    # ISO-8601 UTC: must parse without explosion.
    from datetime import datetime
    datetime.fromisoformat(cfg["started_at"])

    # best.meta.json should exist (a best.pt was saved at first val cycle).
    meta_path = run_dir / "best.meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    for k in ("step", "epoch", "val_loss", "psnr_dark", "saved_at", "started_at", "git_sha"):
        assert k in meta, f"best.meta.json missing key {k!r}"
    # Sidecar's started_at should match config.json's started_at.
    assert meta["started_at"] == cfg["started_at"]
