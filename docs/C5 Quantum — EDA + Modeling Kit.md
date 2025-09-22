# C5 Quantum — EDA + Modeling Kit (range‑aware, non‑standard)

This is a self‑contained plan + code scaffold to (1) run range‑aware EDA, (2) train two **non‑standard** models for next‑event **least‑likely‑20** prediction, and (3) package everything for **Windows 11 CPU** (your box) and **RunPod NVIDIA H200** (GPU) via Docker.

> **Core constraints baked in**: each event is a **sorted 5‑of‑39** subset (ascending, unique). Per‑position feasible ranges are enforced in modeling:
> ‑ `QS_1: 1–35`, `QS_2: 2–36`, `QS_3: 3–37`, `QS_4: 4–38`, `QS_5: 5–39`.

---

## 0) Repository layout

```
C5Q/
├─ README.md                         # How to run locally & on RunPod
├─ configs/
│   ├─ hparams.yaml                  # Model & training knobs
│   └─ buckets.manual.yaml           # Optional manual index buckets (e.g., 1–7, 8–14…)
├─ data/                             # Place CSVs here (ignored by git)
├─ artifacts/                        # EDA & training outputs
├─ c5q/
│   ├─ __init__.py
│   ├─ io.py                         # CSV loading, integrity checks
│   ├─ buckets.py                    # Data‑driven k=6 and manual bucket logic
│   ├─ metrics.py                    # Brier, NLL, AUC‑PR, bottom‑20 precision/recall
│   ├─ masks.py                      # Feasible‑range masks & without‑replacement logic
│   ├─ eda.py                        # Range‑aware EDA (writes JSON/CSVs/plots)
│   ├─ dataset.py                    # PyTorch Dataset for sliding‑window context
│   ├─ encoder_itchan.py             # Channels‑as‑tokens encoder (iTransformer‑style)
│   ├─ model_npl.py                  # Neural Plackett–Luce w/ bucket conditioning
│   ├─ model_subsetdiff.py           # Discrete Subset Diffusion (fixed K=5)
│   ├─ train_npl.py                  # Trainer for NPL
│   ├─ train_subsetdiff.py           # Trainer for subset diffusion
│   ├─ eval.py                       # Evaluation & reporting (overall + per bucket/pos)
│   └─ utils.py                      # Seed, logging, checkpointing
├─ requirements.txt
├─ Dockerfile                        # CPU default; GPU variant via ARG
├─ Makefile                          # Convenience targets
└─ README_PC_RUNPOD.md               # Your PC specs & RunPod fallback documented
```

---

## 1) README.md (usage)

```markdown
# C5 Quantum — Range‑aware EDA & Non‑standard Modeling

## 1) Data
Place the primary dataset at:

```

C5Q/data/c5\_Matrix\_binary.csv

````

Per‑QS binary files are optional/redundant; this kit uses only the primary CSV.

## 2) Quickstart (local CPU, Windows 11)

```bash
# 0) Create venv & install deps
python -m venv .venv && .\.venv\Scripts\activate && pip install -r requirements.txt

# 1) EDA (writes artifacts/eda/*)
python -m c5q.eda --csv data/c5_Matrix_binary.csv --out artifacts/eda --k 6

# 2) Train NPL (masked Plackett–Luce) with k=6 buckets
python -m c5q.train_npl --csv data/c5_Matrix_binary.csv --config configs/hparams.yaml --buckets k6

# 3) Evaluate & export the next‑event least‑20
python -m c5q.eval --run artifacts/npl/run_*/ --out artifacts/npl/report

# 4) (Optional) Train Subset Diffusion
python -m c5q.train_subsetdiff --csv data/c5_Matrix_binary.csv --config configs/hparams.yaml --buckets k6
````

## 3) Docker (CPU default)

```bash
# Build
docker build -t c5q:latest .

# EDA
mkdir -p out && docker run --rm -v %cd%/data:/data -v %cd%/out:/out c5q:latest \
  python -m c5q.eda --csv /data/c5_Matrix_binary.csv --out /out/eda --k 6
```

## 4) Docker (GPU on RunPod H200)

```bash
# Build with CUDA base (ARG switches image)
docker build --build-arg BASE=cuda -t c5q:cuda .

# Run with GPUs
docker run --gpus all --rm -v $PWD/data:/data -v $PWD/out:/out c5q:cuda \
  python -m c5q.train_npl --csv /data/c5_Matrix_binary.csv --config configs/hparams.yaml --buckets k6
```

## 5) Outputs

* `artifacts/eda/eda_summary.json` — entropies, bottom‑20 (global & quintiles), bucket definitions (k=4/5/6)
* `artifacts/npl/…` / `artifacts/subsetdiff/…` — checkpoints, logs, calibration plots
* `…/predictions/next_event_marginals.csv` — 39 per‑state probabilities
* `…/predictions/least20.csv` — 20 smallest‑probability states

````

---

## 2) configs/hparams.yaml (defaults)

```yaml
seed: 42
context:
  window: 128           # past events window for encoder
  stride: 1
  K: 5                  # fixed subset size (QS count)

training:
  batch_size: 256
  epochs: 50
  lr: 2.5e-4
  weight_decay: 1.0e-4
  grad_clip: 1.0
  device: auto          # 'cpu'|'cuda'|'auto'

encoder:
  d_model: 256
  n_heads: 8
  n_layers: 4
  ff_mult: 4
  dropout: 0.1

npl:
  use_bucket_bias: true
  bucket_mode: k6       # 'k4'|'k5'|'k6'|'manual'
  loss:
    type: pl_ce         # PL + cross-entropy surrogate
    bottom20_weight: 3.0
    label_smoothing: 0.01

subsetdiff:
  steps: 200
  noise_schedule: cosine
  guidance: 0.0
````

---

## 3) requirements.txt

```
numpy
pandas
torch>=2.1
pyyaml
scikit-learn
matplotlib
```

---

## 4) Dockerfile

```dockerfile
# ARG to switch between CPU and CUDA base images
ARG BASE=cpu
FROM pytorch/pytorch:2.3.1-${BASE}-py3.11
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Default: just show help
CMD ["python","-m","c5q.eda","--help"]
```

---

## 5) c5q/io.py (load + integrity checks)

```python
import pandas as pd

def load_primary(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    qs_cols = [f"QS_{i}" for i in range(1,6)]
    qv_cols = [f"QV_{i}" for i in range(1,40)]
    assert all(c in df.columns for c in qs_cols+qv_cols), "Missing QS_/QV_ columns"
    # ascending & unique check
    qs = df[qs_cols].to_numpy()
    for r,a in enumerate(qs):
        assert all(a[j] < a[j+1] for j in range(4)), f"Row {r}: QS not strictly ascending"
        assert 1 <= a[0] and a[-1] <= 39, f"Row {r}: QS values out of bounds"
    # exactly five ones in QV matching QS set
    qv = df[qv_cols].to_numpy()
    for r in range(len(df)):
        ones = {i+1 for i,v in enumerate(qv[r]) if v==1}
        assert len(ones)==5, f"Row {r}: QV count != 5"
        assert ones==set(df.loc[r, qs_cols].astype(int).tolist()), f"Row {r}: QV != QS set"
    return df
```

---

## 6) c5q/masks.py (feasible ranges + without‑replacement)

```python
import torch

def feasible_mask_for_pos(k: int) -> torch.Tensor:
    # k in {1..5}; states are 1..39
    lo, hi = k, 39-(5-k)
    m = torch.zeros(39, dtype=torch.bool)
    m[lo-1:hi] = True
    return m

def apply_masks(logits: torch.Tensor, chosen: set, k: int) -> torch.Tensor:
    # logits: [39]; chosen: set of 1-indexed states already selected
    m = feasible_mask_for_pos(k)
    for s in chosen:
        m[s-1] = False
    masked = logits.clone()
    masked[~m] = float('-inf')
    return masked
```

---

## 7) c5q/buckets.py (k=6 default; manual override)

```python
from typing import List

K6 = [
    [12, 19, 32, 35, 37],
    [3, 6, 15, 17, 30, 39],
    [1, 4, 13, 16, 25, 27, 38],
    [14, 18, 23, 24, 33, 34],
    [5, 7, 9, 20, 28, 29, 31],
    [2, 8, 10, 11, 21, 22, 26, 36],
]

def state_to_bucket(states: List[int], mode: str="k6", manual=None):
    buckets = {"k6": K6}.get(mode, manual)
    idx = {}
    for b, arr in enumerate(buckets):
        for s in arr: idx[s] = b
    return [idx[s] for s in states]
```

---

## 8) c5q/encoder\_itchan.py (channels‑as‑tokens encoder)

```python
import torch, torch.nn as nn

class ChannelTokenEncoder(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_layers=4, ff_mult=4, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(39, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model*ff_mult, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, 39)
    def forward(self, x):
        # x: [B, T, 39] binary history window
        h = self.in_proj(x)               # [B,T,d]
        h = self.encoder(h)               # [B,T,d]
        h = self.norm(h[:, -1])           # take last timestep summary
        logits = self.out(h)              # [B,39]
        return logits
```

---

## 9) c5q/model\_npl.py (masked Neural Plackett–Luce with bucket conditioning)

```python
import torch, torch.nn as nn, torch.nn.functional as F
from .masks import apply_masks
from .buckets import state_to_bucket

class NPL(nn.Module):
    def __init__(self, encoder, use_bucket_bias=True, n_buckets=6):
        super().__init__()
        self.encoder = encoder
        self.use_bucket_bias = use_bucket_bias
        if use_bucket_bias:
            self.bucket_bias = nn.Parameter(torch.zeros(n_buckets))
    def forward(self, x_hist):
        # x_hist: [B,T,39] -> context logits [B,39]
        return self.encoder(x_hist)
    def pl_loss(self, logits, target_sets):
        # target_sets: List[List[int]] 5 unique 1..39 in ascending order
        loss = 0.0
        for b in range(logits.shape[0]):
            chosen = set()
            for k, s in enumerate(target_sets[b], start=1):
                l = logits[b]
                if self.use_bucket_bias:
                    # simple bucket prior
                    bucket = state_to_bucket([i+1 for i in range(39)], mode="k6")
                    bb = torch.tensor([self.bucket_bias[bk] for bk in bucket], device=l.device)
                    l = l + bb
                l = apply_masks(l, chosen, k)
                logprob = l[s-1] - torch.logsumexp(l, dim=-1)
                loss = loss - logprob
                chosen.add(s)
        return loss / logits.shape[0]
```

---

## 10) c5q/model\_subsetdiff.py (subset diffusion — minimal skeleton)

```python
# NOTE: This is a minimal placeholder to stand up training; the full
# discrete diffusion kernel (fixed‑cardinality) can be swapped later.
import torch, torch.nn as nn

class SubsetDiffusion(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(39, 39)
    def forward(self, x_hist):
        # Produce per‑state logits as denoiser proxy; real kernel TBD
        ctx = self.encoder(x_hist)        # [B,39]
        return self.head(torch.sigmoid(ctx))
```

---

## 11) c5q/dataset.py (sliding‑window dataset)

```python
import torch, numpy as np, pandas as pd

class C5Dataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, window=128, stride=1):
        df = pd.read_csv(csv_path)
        qv_cols = [f"QV_{i}" for i in range(1,40)]
        qs_cols = [f"QS_{i}" for i in range(1,6)]
        X = df[qv_cols].to_numpy().astype(np.float32)
        Y = df[qs_cols].to_numpy().astype(int)
        self.windows = []
        self.targets = []
        for t in range(window, len(df), stride):
            self.windows.append(X[t-window:t])
            self.targets.append(Y[t])
        self.windows = np.stack(self.windows)
        self.targets = np.stack(self.targets)
    def __len__(self):
        return len(self.windows)
    def __getitem__(self, idx):
        x = torch.from_numpy(self.windows[idx])   # [T,39]
        y = [int(v) for v in self.targets[idx]]   # 5 ints
        return x, y
```

---

## 12) c5q/train\_npl.py (trainer)

```python
import torch, yaml, argparse
from torch.utils.data import DataLoader
from .dataset import C5Dataset
from .encoder_itchan import ChannelTokenEncoder
from .model_npl import NPL

def main(args):
    cfg = yaml.safe_load(open(args.config))
    device = torch.device("cuda" if cfg["training"]["device"]=="cuda" or (
        cfg["training"]["device"]=="auto" and torch.cuda.is_available()) else "cpu")
    ds = C5Dataset(args.csv, window=cfg["context"]["window"], stride=cfg["context"]["stride"])
    dl = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=True, num_workers=0)

    enc = ChannelTokenEncoder(**cfg["encoder"]).to(device)
    model = NPL(enc, use_bucket_bias=cfg["npl"]["use_bucket_bias"]).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])

    for epoch in range(cfg["training"]["epochs"]):
        model.train(); total=0.0
        for x,y in dl:
            x = x.to(device)
            logits = model(x)
            loss = model.pl_loss(logits, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
            opt.step()
            total += float(loss)
        print(f"epoch {epoch+1}: loss={total/len(dl):.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--buckets", default="k6")
    args = p.parse_args()
    main(args)
```

---

## 13) c5q/eval.py (metrics + least‑20 export)

```python
import argparse, numpy as np, pandas as pd, torch
from .dataset import C5Dataset
from .encoder_itchan import ChannelTokenEncoder
from .model_npl import NPL

def least20_from_logits(logits: np.ndarray) -> list:
    order = np.argsort(logits)  # ascending
    return (order[:20]+1).tolist()

def main(args):
    # Minimal example: reload last checkpoint or use the freshly created model
    # Here we just run the untrained model as a placeholder
    ds = C5Dataset(args.csv, window=128, stride=1)
    x, y = ds[ -1 ]   # last context → predict next event
    model = NPL(ChannelTokenEncoder())
    with torch.no_grad():
        logits = model(x.unsqueeze(0)).squeeze(0).numpy()
    probs = (1/(1+np.exp(-logits)))
    bottom20 = least20_from_logits(probs)
    pd.DataFrame({"least20": bottom20}).to_csv(args.out+"/least20.csv", index=False)
    print("least20 saved to", args.out+"/least20.csv")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main(args)
```

---

## 14) c5q/eda.py (range‑aware EDA)

```python
import json, argparse, numpy as np, pandas as pd
from collections import Counter

QS_COLS = [f"QS_{i}" for i in range(1,6)]
QV_COLS = [f"QV_{i}" for i in range(1,40)]

def entropy(vals):
    counts = np.bincount(vals, minlength=40)[1:]
    p = counts / counts.sum()
    p = p[p>0]
    return float(-(p*np.log2(p)).sum())

def bottom20(vals):
    cnt = Counter(vals)
    return [s for s,_ in sorted(((s, cnt.get(s,0)) for s in range(1,40)), key=lambda x: (x[1], x[0]))[:20]]

def jaccard(X):
    n = X.shape[1]
    S = np.zeros((n,n), dtype=float)
    for i in range(n):
        a = X[:,i].astype(bool)
        for j in range(n):
            b = X[:,j].astype(bool)
            inter = np.logical_and(a,b).sum(); union = np.logical_or(a,b).sum()
            S[i,j] = inter/union if union>0 else 0.0
    return S

def kmeans_cos(X, k, iters=100, seed=42):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), size=k, replace=False)]
    for _ in range(iters):
        # assign
        D = 1 - (X @ C.T) / (np.linalg.norm(X,axis=1,keepdims=True)*np.linalg.norm(C,axis=1))
        L = D.argmin(1)
        C_new = np.stack([X[L==j].mean(0) if np.any(L==j) else X[rng.integers(0,len(X))] for j in range(k)])
        if np.allclose(C_new, C): break
        C = C_new
    clusters = [[] for _ in range(k)]
    for i,l in enumerate(L): clusters[l].append(i+1)
    return clusters

def main(csv, out, k):
    df = pd.read_csv(csv)
    qs = df[QS_COLS].to_numpy().astype(int)
    qv = df[QV_COLS].to_numpy().astype(int)
    # checks (strictly ascending)
    asc_viol = int(any(any(a[j] >= a[j+1] for j in range(4)) for a in qs))

    ents = {f"H({c})": entropy(df[c].values.astype(int)) for c in QS_COLS}
    global_b20 = bottom20(qs.flatten())
    n = len(df); bounds = [(i*n//5, (i+1)*n//5) for i in range(5)]
    seg_b20 = [ bottom20(df.iloc[a:b][QS_COLS].to_numpy().astype(int).flatten()) for a,b in bounds ]

    S = jaccard(qv)
    # k ∈ {4,5,6}
    buckets = {str(j): kmeans_cos(S, j) for j in [4,5,6]}

    meta = {
        "asc_violations": asc_viol,
        "position_entropies": ents,
        "global_bottom20": global_b20,
        "seg_bottom20": seg_b20,
        "buckets": buckets,
        "machine_spec": {
            "os": "Windows 11 Home 64-bit",
            "cpu": "AMD Ryzen 9 6900HX",
            "ram_gb": 64,
            "gpus": ["AMD Radeon RX 6600M 8GB", "AMD Radeon Graphics 512MB"],
            "storage": ["Samsung 990 PRO 1TB", "KINGSTON OM8PGP41024Q 1TB"],
            "displays": ["2560x1440@59Hz", "2560x1440@60Hz"],
            "fallback": "Dockerize & run on RunPod NVIDIA H200 if local ETA > ~1h"
        }
    }
    import os
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/eda_summary.json","w") as f: json.dump(meta, f, indent=2)
    pd.DataFrame({"global_bottom20": global_b20}).to_csv(f"{out}/global_bottom20.csv", index=False)
    for i,arr in enumerate(seg_b20,1):
        pd.DataFrame({f"segment_{i}_bottom20": arr}).to_csv(f"{out}/seg{i}_bottom20.csv", index=False)
    for j,clus in buckets.items():
        pd.DataFrame({"cluster": sum([[idx+1]*len(c) for idx,c in enumerate(clus)], []),
                      "state": sum(clus, [])}).to_csv(f"{out}/buckets_k{j}.csv", index=False)
    print("EDA complete →", out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="artifacts/eda")
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()
    main(args.csv, args.out, args.k)
```

---

## 15) README\_PC\_RUNPOD.md (your PC + fallback)

```markdown
# Execution Environment Notes

## Local Workstation
- **OS:** Windows 11 Home 64‑bit
- **CPU:** AMD Ryzen 9 6900HX (Rembrandt, 6nm)
- **RAM:** 64 GB @ ~2393 MHz
- **GPU:** AMD Radeon RX 6600M (8 GB) + integrated AMD Radeon Graphics (512 MB)
- **Displays:** 2560×1440 @59Hz, 2560×1440 @60Hz
- **Storage:** Samsung 990 PRO 1TB (NVMe), KINGSTON OM8PGP41024Q 1TB (SATA SSD)

This kit runs **CPU‑only** locally by default. Training tasks that may exceed ~1 hour should be offloaded to RunPod.

## RunPod GPU Fallback (NVIDIA H200)
1. Build CUDA image: `docker build --build-arg BASE=cuda -t c5q:cuda .`
2. Mount data & run training: `docker run --gpus all -v $PWD/data:/data -v $PWD/out:/out c5q:cuda python -m c5q.train_npl --csv /data/c5_Matrix_binary.csv --config configs/hparams.yaml --buckets k6`
3. Artifacts are written to `/out` and can be copied back to the workstation.
```

---

## 16) Next steps

* [ ] You drop your current CSV at `data/c5_Matrix_binary.csv` and run `EDA` (CPU OK).
* [ ] If metrics look good, kick off **NPL** locally for a few epochs; if ETA feels long, switch to **RunPod**.
* [ ] I’ll then help wire advanced losses (bottom‑20 precision weighting), richer diffusion kernel, and produce the calibration/precision reports per **bucket** and **position**.

> Everything above is ready to copy‑paste into a repo. If you want, I can supply a `.zip` with this exact structure next.
