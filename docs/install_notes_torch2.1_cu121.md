# Installation Notes — torch 2.1 + CUDA 12.1

Supplementary notes for installing YOLO-World on a modern host where the
official `docs/installation.md` recipe hits multiple pitfalls.

These notes document a **working** install as of 2026-04 on:

- Ubuntu (Linux 6.8), NVIDIA driver 550.144.03 (CUDA runtime 12.4 supported)
- Python 3.10, conda env
- torch 2.1.2 + cu121 prebuilt wheels

They also point out several upstream issues worth fixing via PR:

1. `pyproject.toml` pins are internally inconsistent.
2. `mmcv` + `mmcv-lite` are both listed and collide on uninstall.
3. `yolo_world/models/detectors/yolo_world.py:61` has a literal `SyntaxError`
   introduced in commit `1f12bca1` (2025-02-27 "add mask").
4. `mmyolo` git dep fails under pip build isolation.
5. setuptools 81+ removes `pkg_resources`, which torch 2.1 still imports.
6. The `lvis` PyPI package is an undeclared runtime dependency.
7. `init_detector` forces the user to have the LVIS annotation file on disk,
   even for text-prompt-only inference where no dataset is needed.

---

## Working environment — version matrix

Exactly these versions load cleanly, with CUDA available, no numpy/ABI warnings,
and `import yolo_world` succeeding:

| Package                | Version           | Source                              | Notes                                          |
|------------------------|-------------------|-------------------------------------|------------------------------------------------|
| python                 | 3.10.20           | conda-forge                         |                                                |
| numpy                  | 1.26.4            | PyPI                                | **Must be `<2`** — torch 2.1 built against 1.x |
| torch                  | 2.1.2+cu121       | download.pytorch.org/whl/cu121      | CUDA available on driver ≥ 535                 |
| torchvision            | 0.16.2+cu121      | download.pytorch.org/whl/cu121      | Matches torch 2.1.2 exactly                    |
| triton                 | 2.1.0             | PyPI (dep of torch)                 |                                                |
| opencv-python          | 4.9.0.80          | PyPI                                | Authors' pin (basic_requirements.txt)          |
| opencv-python-headless | 4.11.0.86         | PyPI (dep of albumentations)        | Newer than authors' 4.2.0.34 but works         |
| mmcv                   | 2.1.0             | openmmlab cu121/torch2.1 wheel      | Full build with CUDA ops                       |
| mmengine               | 0.10.7            | PyPI (via openmim)                  |                                                |
| mmdet                  | **3.3.0**         | PyPI                                | **Bumped from pyproject's 3.0.0** (see below)  |
| mmyolo                 | 0.6.0             | git+onuralpszr/mmyolo fork          | `--no-build-isolation` required                |
| transformers           | 4.36.2            | PyPI                                | Authors' pin                                   |
| tokenizers             | 0.15.2            | PyPI (dep of transformers 4.36.2)   |                                                |
| huggingface-hub        | 0.36.2            | PyPI                                |                                                |
| timm                   | 0.6.13            | PyPI                                | Authors' pin                                   |
| albumentations         | 2.0.8             | PyPI                                |                                                |
| supervision            | 0.19.0            | PyPI                                | Authors' pin                                   |
| setuptools             | 79.0.1 at build   | PyPI                                | Downgraded to 60.2.0 at runtime by openxlab    |
| yolo_world             | 0.1.0 (editable)  | `pip install -e .`                  |                                                |

Smoke test — the following must all succeed and print versions:

```python
import numpy, torch, torchvision, cv2
import mmcv, mmengine, mmdet, mmyolo
import transformers, timm, albumentations, supervision
import yolo_world
from yolo_world.models.detectors import YOLOWorldDetector
from mmcv.ops import nms  # confirms mmcv CUDA ops compiled
assert torch.cuda.is_available()
```

---

## Known issues and fixes

### Issue 1 — `pyproject.toml` version pins are inconsistent

The current `pyproject.toml` lists:

```toml
dependencies = [
    ...
    "torchvision>=0.16.2",
    "mmcv-lite>=2.0.0rc4",
    "mmdet==3.0.0",
    "mmcv",
    'mmyolo @ git+https://github.com/onuralpszr/mmyolo.git',
]
```

Problems:

- `torchvision>=0.16.2` is unbounded. On a fresh env pip will pick the latest
  torchvision (currently 0.26.0+), which pulls a torch ≥ 2.8 as a transitive,
  and that breaks any mmcv prebuilt wheel we already installed.
- `mmdet==3.0.0` asserts `mmcv>=2.0.0rc4, <2.1.0` at import time. But the
  `mmcv` prebuilt wheels for torch 2.1 / cu121 start at **mmcv 2.1.0** — so
  there is no prebuilt combination that satisfies both pins. The options are:
  1. Compile `mmcv==2.0.x` from source (slow, fragile; needs nvcc).
  2. Bump `mmdet` to a version whose assertion window includes 2.1.0 — e.g.
     `mmdet==3.3.0` (asserts `<2.2.0`).
- `mmyolo` is installed from a git fork. Its `setup.py` imports `torch` at
  build time, which fails under pip's default build isolation
  (`ModuleNotFoundError: No module named 'torch'`). Users must pass
  `--no-build-isolation` or the project's top-level
  `pyproject.toml` should declare `torch` in `[build-system].requires`.

**Proposed upstream fix (PR-ready):**

```toml
dependencies = [
    "wheel",
    "torch>=2.1.0,<2.2",
    "torchvision>=0.16.0,<0.17",
    "transformers==4.36.2",
    "tokenizers",
    "numpy<2",
    "opencv-python==4.9.0.80",
    "supervision==0.19.0",
    "openmim",
    "mmcv>=2.1.0,<2.2",
    "mmdet>=3.3.0,<3.4",
    "mmengine>=0.10.3",
    'mmyolo @ git+https://github.com/onuralpszr/mmyolo.git',
    "timm==0.6.13",
    "albumentations",
    "lvis",
]
```

Key differences from the current file:

- Drop `mmcv-lite` entirely (see Issue 2).
- Upper-bound `torch`/`torchvision` to avoid pip picking a newer torch that
  breaks mmcv.
- Bump `mmdet` to `>=3.3.0` to allow `mmcv 2.1.x` prebuilts.
- Add `numpy<2` (torch 2.1 is not NumPy-2 compatible).
- Move `timm` and `albumentations` out of `requirements/basic_requirements.txt`
  and into `pyproject.toml` so `pip install -e .` is self-sufficient.
- Add `lvis` so every demo (which routes through `init_detector` → LVIS
  test dataset) has the parser it needs (see Issue 6).

### Issue 2 — `mmcv` and `mmcv-lite` collide

`mmcv` (full, with CUDA ops) and `mmcv-lite` (Python-only, no CUDA ops) are
distinct PyPI distributions but they install the **same** `mmcv/` import
namespace. When both are installed, one silently shadows the other. Worse,
uninstalling one removes shared files from the other, leaving a broken
namespace (`module 'mmcv' has no attribute '__version__'`).

The current `pyproject.toml` lists both:

```toml
"mmcv-lite>=2.0.0rc4",
...
"mmcv",
```

This near-guarantees that every fresh install ends up with the lite version
shadowing the full one. On `import mmdet` you then see:

```
AssertionError: MMCV==2.2.0 is used but incompatible.
Please install mmcv>=2.0.0rc4, <2.1.0.
```

...reporting the **lite** version even though the full one is also installed.

**Fix:** remove `mmcv-lite` from the dependency list. YOLO-World depends on
CUDA ops via mmcv, so the full build is required anyway.

**Recovery if you hit this:** after `pip uninstall mmcv-lite`, the full `mmcv`
on disk will be half-deleted. Repair with:

```bash
pip install --force-reinstall --no-deps \
  mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
```

The same trap applies to `opencv-python` and `opencv-python-headless`: they
share the `cv2/` namespace. Removing one corrupts the other. If your `cv2`
starts throwing `AttributeError: module 'cv2' has no attribute 'COLOR_BGR2RGB'`,
force-reinstall the one you want to keep.

### Issue 3 — `SyntaxError` in `yolo_world.py` line 61

In `yolo_world/models/detectors/yolo_world.py`, the `reparameterize()` method
contains:

```python
def reparameterize(self, texts: List[List[str]]) -> None:
    # encode text embeddings into the detector
    self.texts = texts
    self.text_feats, None = self.backbone.forward_text(texts)   # SyntaxError
```

`None` is a keyword and cannot appear on the left-hand side of an assignment.
Importing `yolo_world` fails with:

```
SyntaxError: cannot assign to None
```

Introduced in commit `1f12bca1` (2025-02-27, "add mask").

Both implementations of `forward_text()` in `yolo_world/models/backbones/mm_backbone.py`
(`PseudoLanguageBackbone.forward_text` at line 172 and
`MultiModalYOLOBackbone.forward_text` at line 234) return a **single tensor**,
not a tuple. All other call sites (lines 163, 168) treat it as a single value.

**Fix — drop the unpack entirely:**

```python
self.text_feats = self.backbone.forward_text(texts)
```

### Issue 4 — `mmyolo` build needs `--no-build-isolation`

The onuralpszr/mmyolo fork's `setup.py` imports `torch` at build time (to
detect CUDA version for extension compilation). Under pip's default build
isolation, the build sandbox does not have torch, so:

```
ModuleNotFoundError: No module named 'torch'
ERROR: Failed to build 'mmyolo' when getting requirements to build wheel
```

**Workaround:** install with `--no-build-isolation` so the build uses the
currently-activated env's torch:

```bash
pip install --no-build-isolation "mmyolo @ git+https://github.com/onuralpszr/mmyolo.git"
pip install --no-build-isolation -e .
```

**Upstream fix:** add `torch` to the top-level `pyproject.toml`'s
`[build-system].requires`. That makes pip pull torch into the isolated build
env. (YOLO-World's current `[build-system].requires` already includes `torch`,
but that only covers the YOLO-World package itself, not its mmyolo git
dependency.) The right fix is in the mmyolo fork.

### Issue 5 — `setuptools` PEP 660 vs `pkg_resources` deprecation

Editable installs (`pip install -e .`) require setuptools ≥ 64 for PEP 660
support. But setuptools 81+ drops `pkg_resources`, which torch 2.1.2's
`cpp_extension.py` still imports:

```python
from pkg_resources import packaging   # torch/utils/cpp_extension.py
```

So any package build that touches `cpp_extension` (e.g. mmyolo) fails under
setuptools 82.

**Resolution:** pin setuptools to the window `>=64,<80`. Setuptools 79.0.1
works. This is a runtime-of-build constraint only; after the install finishes,
other packages (openxlab) may pull setuptools back to 60.2.0 and that's fine
for runtime use.

### Issue 6 — `lvis` is an undeclared runtime dependency

The default pretrain configs (e.g.
`configs/pretrain/yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py`)
use mmdet's `YOLOv5LVISV1Dataset` as the test dataloader. When you load such a
config via `init_detector(...)` — which every demo in `demo/` does — mmdet's
LVIS loader does `import lvis` at runtime to parse the annotation file. The
`lvis` package is published on PyPI but is **not listed** in `pyproject.toml`
or `requirements/basic_requirements.txt`, so a fresh install hits:

```
ModuleNotFoundError: No module named 'lvis'
...
ImportError: Package lvis is not installed.
Please run "pip install git+https://github.com/lvis-dataset/lvis-api.git".
```

**Workaround:** `pip install lvis` (the PyPI package works; no need for the
git URL in the error message).

**Upstream fix:** add `lvis` to the `dependencies` list in `pyproject.toml`.
It's tiny (~14 KB) and pulls only `Cython` as a new transitive.

### Issue 7 — `init_detector` forces the user to own a dataset for text-only inference

Even after `pip install lvis`, `init_detector` still fails on a clean checkout:

```
FileNotFoundError: [Errno 2] No such file or directory:
'data/coco/lvis/lvis_v1_minival_inserted_image_name.json'
```

Root cause is in mmdet 3.x: `init_detector` builds the test dataset purely to
read `dataset.metainfo`, but mmengine's `BaseDataset.__init__` calls
`full_init()` unconditionally, which calls `load_data_list()` → opens
`ann_file` → fails if the file is absent. In other words, even though the
demo uses its own `texts=[['power socket'], [' ']]` prompts and never looks at
a single LVIS ground-truth annotation, you **cannot build the model** without
having the LVIS annotation JSON on disk at the path the config expects.

Every YOLO-World demo in `demo/` (`simple_demo.py`, `image_demo.py`,
`video_demo.py`, `gradio_demo.py`, etc.) goes through `init_detector`, so
every user who wants to try text-prompt inference first has to download the
entire LVIS annotation set just to pass this init hurdle. It is probably the
single biggest papercut for first-time users.

**Short-term workaround — a minimal stub JSON that satisfies the loader:**

```bash
mkdir -p data/coco/lvis
cat > data/coco/lvis/lvis_v1_minival_inserted_image_name.json <<'JSON'
{"info": {"description": "stub for inference-only demo"}, "licenses": [],
 "images": [{"id": 1, "width": 640, "height": 480, "license": 0,
             "flickr_url": "", "coco_url": "", "date_captured": "",
             "file_name": "stub.jpg", "not_exhaustive_category_ids": [],
             "neg_category_ids": []}],
 "annotations": [],
 "categories": [{"id": 1, "name": "object", "synset": "object.n.01",
                 "synonyms": ["object"], "def": "placeholder",
                 "instance_count": 0, "image_count": 0, "frequency": "f",
                 "synset_parent": ""}]}
JSON
```

Note:
- `images` must contain **at least one entry** — mmengine's
  `_serialize_data()` calls `np.concatenate(data_list)` and errors on an empty
  list (`ValueError: need at least one array to concatenate`).
- The dummy image does **not** need to exist on disk. Only the annotation JSON
  is parsed; no image file is ever read during `init_detector`.
- `categories` needs at least one entry, but its contents are irrelevant —
  the demo replaces `metainfo.classes` with its own `texts=` at inference time.

**Proposed upstream fixes (in order of preference):**

1. **Ship a tiny `configs/inference/` variant** that uses a class-list-based
   dataset (e.g. `mmdet.CocoDataset` with `metainfo=dict(classes=('object',))`
   and `lazy_init=True`), so demo users never touch LVIS. The demo scripts
   would default to this config.
2. **Bypass `init_detector` entirely in the demos.** Instead of calling
   `init_detector(cfg, checkpoint=...)`, build the model directly:
   ```python
   from mmengine.registry import MODELS
   from mmengine.runner import load_checkpoint
   model = MODELS.build(cfg.model)
   load_checkpoint(model, checkpoint, map_location=device)
   model.to(device).eval()
   ```
   This skips `DATASETS.build(test_dataset_cfg)` entirely. The only thing
   demos currently get from `init_detector` beyond model construction is
   dataset `metainfo`, which they don't actually use because `texts=` is
   injected per-call.
3. **Upstream at mmdet:** add an opt-in `init_metainfo_only=True` path to
   `init_detector` that reads `metainfo` from a lightweight config dict
   instead of building the full dataset. (Lowest-odds fix — mmdet moves
   slowly.)

Option 1 is the simplest, most user-friendly PR and lives entirely inside this
repo; option 2 is the most principled; option 3 is the cleanest but least
likely to land.

---

## Reference install recipe (what actually worked)

Assuming a fresh conda env with Python 3.10 and an NVIDIA driver new enough
for CUDA 12.1 runtime:

```bash
# 1. Create env
conda create -n yolow python=3.10 -y
conda activate yolow

# 2. Torch 2.1.2 + cu121 (matching torchvision)
pip install torch==2.1.2 torchvision==0.16.2 \
  --index-url https://download.pytorch.org/whl/cu121

# 3. mmcv 2.1.0 via openmim (prebuilt wheel for torch 2.1 / cu121)
pip install openmim
mim install "mmcv==2.1.0"

# 4. Pin numpy before it drifts to 2.x
pip install "numpy<2"

# 5. Setuptools in the PEP 660 ↔ pkg_resources sweet spot
pip install "setuptools>=64,<80"

# 6. mmyolo fork, no build isolation (so it can see torch)
pip install --no-build-isolation \
  "mmyolo @ git+https://github.com/onuralpszr/mmyolo.git"

# 7. Project install (will pull mmdet, transformers, supervision, etc.)
pip install --no-build-isolation -e .

# 8. Fix the resolver's regressions: undo mmcv-lite shadow, pin proper versions
pip uninstall -y mmcv-lite opencv-python-headless
pip install --force-reinstall --no-deps \
  mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
pip install --force-reinstall --no-deps "opencv-python==4.9.0.80"
pip install "numpy<2" "transformers==4.36.2" "timm==0.6.13" \
            albumentations "mmdet==3.3.0"

# 9. Apply the one-line yolo_world.py fix (Issue 3)
sed -i 's/self\.text_feats, None = /self.text_feats = /' \
  yolo_world/models/detectors/yolo_world.py

# 10. Undeclared runtime dep for every demo (Issue 6)
pip install lvis

# 11. Stub LVIS annotation file so init_detector stops demanding the
#     real dataset (Issue 7). The contents don't matter — no image is read.
mkdir -p data/coco/lvis
cat > data/coco/lvis/lvis_v1_minival_inserted_image_name.json <<'JSON'
{"info": {"description": "stub for inference-only demo"}, "licenses": [],
 "images": [{"id": 1, "width": 640, "height": 480, "license": 0,
             "flickr_url": "", "coco_url": "", "date_captured": "",
             "file_name": "stub.jpg", "not_exhaustive_category_ids": [],
             "neg_category_ids": []}],
 "annotations": [],
 "categories": [{"id": 1, "name": "object", "synset": "object.n.01",
                 "synonyms": ["object"], "def": "placeholder",
                 "instance_count": 0, "image_count": 0, "frequency": "f",
                 "synset_parent": ""}]}
JSON
```

If the proposed `pyproject.toml` changes in Issue 1 land upstream, steps 4–10
collapse into a single clean `pip install --no-build-isolation -e .`. Step 11
(the LVIS stub) only goes away if one of the Issue 7 upstream fixes lands.

---

## Pitfalls unrelated to the repo

These bit the install but are environment-specific, not YOLO-World's fault.
Mentioning them so future installers don't lose an afternoon:

- **ROS 2 sourced in `~/.bashrc`** injects `/opt/ros/humble/...` into
  `PYTHONPATH`, which leaks into every conda env. Pip then sees ROS packages
  as "installed" in the env and may uninstall or "upgrade" them. Mitigate by
  moving `source /opt/ros/humble/setup.bash` out of `.bashrc` into a shell
  function (`ros2-on`) that you invoke only when you need ROS.
- **`~/.local/lib/python3.x/site-packages`** is on `sys.path` in every conda
  env by default. Pip will happily uninstall/upgrade packages living there
  while working on an "isolated" env. Set `PYTHONNOUSERSITE=1` for the env
  (conda activation hook) to make Python ignore `~/.local`.

Both of these caused real collateral damage during the install session that
produced these notes (torch in `~/.local` got silently uninstalled when pip
installed torch into the conda env).

---

## Personal branch workflow (amrutur's fork)

> **Note:** This section is specific to the `personal/working-notes` branch
> on `github.com/amrutur/YOLO-World`. It documents how to sync this branch
> across machines and keep it up to date with upstream. It is NOT relevant
> to anyone sending a PR upstream — those would go through `master` on a
> clean fork.

### Repository topology

```
AILab-CVC/YOLO-World  (upstream, no push access)
    ↑
    │
amrutur/YOLO-World    (my fork)
    ├── master                          mirrors upstream, kept clean for future PRs
    ├── fix/reparameterize-syntaxerror  one-commit bug fix → upstream PR #652
    └── personal/working-notes          this branch — fix + docs + CLAUDE.md + LVIS stub
```

The `personal/working-notes` branch is based off `fix/reparameterize-syntaxerror`,
so the `import yolo_world` SyntaxError is pre-fixed. The two commits on top of
upstream master are:

1. the one-line SyntaxError fix in `yolo_world/models/detectors/yolo_world.py:61`
2. the working-tree notes commit (this file, architecture walkthrough, CLAUDE.md,
   LVIS stub, and branch-local `.gitignore` additions).

### Clone + check out on a fresh machine

```bash
git clone https://github.com/amrutur/YOLO-World.git
cd YOLO-World
git checkout personal/working-notes
```

Then follow the "Reference install recipe" earlier in this file to build the
conda env. Download the model weights from HuggingFace (links in the README);
`weights/` is gitignored on this branch so the checkpoints do not ride along.
The LVIS annotation stub at `data/coco/lvis/lvis_v1_minival_inserted_image_name.json`
IS tracked, so `init_detector` works out of the box for text-prompt-only
inference.

### Adding more notes / updating the branch

When adding more local docs, edits, experiments that you want backed up:

```bash
git checkout personal/working-notes
# make edits
git add <files>                             # or `git add -f <file>` if gitignored
git commit -m "notes: <what you changed>"
git push amrutur personal/working-notes
git checkout master                         # back to master when done
```

### Keeping `personal/working-notes` in sync with upstream

When `AILab-CVC/YOLO-World` gets new commits and you want to rebase your
notes on top:

```bash
# 1. Pull upstream into your local master
git fetch origin
git checkout master
git merge --ff-only origin/master

# 2. Push the updated master to your fork (optional but tidy)
git push amrutur master

# 3. Rebase the fix branch on top of upstream master
git checkout fix/reparameterize-syntaxerror
git rebase master
git push --force-with-lease amrutur fix/reparameterize-syntaxerror

# 4. Rebase personal/working-notes on top of the updated fix branch
git checkout personal/working-notes
git rebase fix/reparameterize-syntaxerror
git push --force-with-lease amrutur personal/working-notes
```

`--force-with-lease` is safe on personal branches that nobody else pulls
from — it refuses to push if someone else has added commits in the
meantime (which should never happen on a personal branch). Never use plain
`--force` on shared branches.

If the upstream SyntaxError fix lands (PR #652), the rebase will see that
the fix branch's commit is already upstream and drop it automatically.
At that point you can simplify by basing `personal/working-notes` directly
off `master` instead of the fix branch.

### Branch-local `.gitignore` vs master `.gitignore`

On this branch, `.gitignore` has extra entries (`weights/`, `demo_outputs/`,
`data/coco/`) that are NOT present on upstream master. Don't be surprised if
those entries vanish when you switch to master — that's expected.

The files kept OUT of version control on this branch (via those rules):

| Pattern | Why excluded |
|---|---|
| `weights/` | 422 MB per checkpoint, re-download from HuggingFace |
| `demo_outputs/` | Regeneratable in seconds from any checkpoint + input image |
| `data/coco/` | Real COCO imagery is huge and user-specific |

The files tracked DESPITE being in an ignored directory (via `git add -f`):

| File | Size | Why tracked |
|---|---|---|
| `data/coco/lvis/lvis_v1_minival_inserted_image_name.json` | < 1 KB | Minimal stub so `init_detector` works without a real LVIS dataset (see Issue 7 above) |

### Keeping this document alive

If you evolve the install further, add new notes to this file on
`personal/working-notes`, commit, and push. The living version of this doc
lives at:

`https://github.com/amrutur/YOLO-World/blob/personal/working-notes/docs/install_notes_torch2.1_cu121.md`
