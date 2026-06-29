# SELENE training on Colab

Copy-paste these cells into a Colab notebook in order. The trainer
auto-resumes from `latest.pt`, so re-running Cell 5 after a disconnect
picks up where it left off.

---

## Cell 1 — Mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## Cell 2 — Clone repo + install deps

Skip the clone if the repo is already at `/content/selene` from earlier
in the same session.

```python
!git clone https://github.com/yaga-07/selene.git /content/selene 2>/dev/null || (cd /content/selene && git pull)
%cd /content/selene
!pip install -q -r requirements.txt
```

---

## Cell 3 — Stage the bundle to local disk

Drive is too slow for the mmap'd reads the Dataset makes thousands of
times per epoch. Copy the 23 GB `patches.npy` to `/content/` once
(~15–25 min depending on Drive throughput). Skip if already staged.

```python
import os, shutil, pathlib, time
BUNDLE_SRC = '/content/drive/MyDrive/selene-colab/patches.npy'
BUNDLE_DST = '/content/patches.npy'
if not pathlib.Path(BUNDLE_DST).exists():
    t = time.time()
    shutil.copy(BUNDLE_SRC, BUNDLE_DST)
    print(f"copied in {time.time()-t:.1f}s, {os.path.getsize(BUNDLE_DST)/1e9:.1f} GB")
else:
    print(f"already staged: {os.path.getsize(BUNDLE_DST)/1e9:.1f} GB")
```

---

## Cell 4 — Pre-flight checks

Validates manifest schema, bundle shape/dtype, FPN templates load, model
forward + loss compute on the device, and disk-space headroom for
checkpoints. Bail out and fix anything that fails before launching the
real run.

```python
!python -m training.preflight \
  --bundle /content/patches.npy \
  --device cuda \
  --runs-dir /content/runs \
  --min-free-gb 10
```

---

## Cell 4.5 — Restore run dir from Drive (fresh runtime only)

If the Colab runtime was reset between sessions, `/content/runs/...` is
gone but the Drive mirror is intact. Pull it back so `latest.pt` is
where the trainer expects it.

```python
import shutil, pathlib
mirror = pathlib.Path('/content/drive/MyDrive/selene-runs/E2_pg_nll')
local = pathlib.Path('/content/runs/E2_pg_nll')
if mirror.exists() and not local.exists():
    shutil.copytree(mirror, local)
    print(f"restored from {mirror}")
```

---

## Cell 5 — Launch training (E2 — PG-NLL from scratch)

- Resumes automatically if `/content/runs/E2_pg_nll/latest.pt` exists.
- Mirrors the run dir to Drive at every epoch end.
- Checkpoints every 200 steps. `best.pt` updates on `val_loss` improvement.
- Sample denoised PNGs at every val eval (default every 1000 steps).

```python
!python -m training.train \
  --bundle /content/patches.npy \
  --loss pg_nll \
  --epochs 80 \
  --batch-size 32 \
  --lr 1e-4 \
  --warmup-steps 500 \
  --val-every-steps 1000 \
  --ckpt-every-steps 200 \
  --val-batches 32 \
  --num-workers 2 \
  --device cuda \
  --run-dir /content/runs/E2_pg_nll \
  --drive-mirror /content/drive/MyDrive/selene-runs/E2_pg_nll
```

---

## Cell 6 (optional) — Watch progress in a separate cell

While Cell 5 is running, this tails the per-step log.

```python
!tail -f /content/runs/E2_pg_nll/train_log.csv
```

For per-eval summaries:

```python
!tail -f /content/runs/E2_pg_nll/val_log.jsonl
```

---

## After training

The Drive mirror is the source of truth. After the run completes:

- `runs/E2_pg_nll/best.pt` — checkpoint at best val_loss
- `runs/E2_pg_nll/latest.pt` — final checkpoint
- `runs/E2_pg_nll/train_log.csv` — per-step (step, epoch, lr, loss, wall_s)
- `runs/E2_pg_nll/val_log.jsonl` — per-eval (step, val_loss, psnr_dark)
- `runs/E2_pg_nll/samples/*.png` — noisy / clean / μ triptychs
- `runs/E2_pg_nll/config.json` — frozen args + git SHA

---

## Ablation variants

To run the MSE baseline (E1) for comparison, swap `--loss` and use a
fresh run dir so it doesn't try to resume from the PG-NLL checkpoint:

```python
!python -m training.train \
  --bundle /content/patches.npy \
  --loss mse \
  --epochs 80 --batch-size 32 --lr 1e-4 \
  --warmup-steps 500 --val-every-steps 1000 --ckpt-every-steps 200 \
  --val-batches 32 --num-workers 2 --device cuda \
  --run-dir /content/runs/E1_mse \
  --drive-mirror /content/drive/MyDrive/selene-runs/E1_mse
```
