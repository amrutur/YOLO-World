# YOLO-World Architecture Walkthrough

*Extracted from a Claude Code conversation (2026-04-11/12).*
*Covers the detector, backbone, neck (VL-PAN), and head with code references and tensor shapes.*

---

<!-- Section: Part 1 — The big picture and the detector -->

Now I have the full detector. Let me write Part 1 — the top-level picture and the detector — and we'll build out from there.

# Part 1 — The big picture and the detector

## The 10,000-foot view

YOLO-World is **YOLOv8 + CLIP text encoder, fused at two points**. The architecture is:

```
                       ┌──────────────────────────┐
  text prompts ──────► │ CLIP text encoder (frozen)│─── txt_feats (B, N, 512) ─┐
    (N strings)        └──────────────────────────┘                            │
                                                                               ▼
                       ┌─────────────────┐      ┌──────────────┐      ┌─────────────────┐
  image (B,3,H,W) ───► │ CSPDarknet      │────► │ VL-PAFPN     │────► │ YOLO-World head │────► boxes, scores
                       │ (YOLOv8 backbone)│  ↑   │ (neck, fuses │  ↑   │ (contrastive    │
                       │ P3,P4,P5         │  │   │ text at each │  │   │ classification  │
                       └─────────────────┘  │   │ stage)       │  │   │ against txt)    │
                                            │   └──────────────┘  │   └─────────────────┘
                                            │                     │
                                            └─── fusion point 1 ──┴── fusion point 2
```

Two fusion points:

1. **Neck fusion** (VL-PAN): text embeddings gate the image feature maps through a **MaxSigmoidAttnBlock** at every P3/P4/P5 level. This lets text steer *where* the detector looks.
2. **Head fusion** (ContrastiveHead): the head's classification logits are computed as a **dot product between per-region image embeddings and text embeddings** instead of a static `Conv2d(..., num_classes)`. This is what makes the model open-vocabulary — change the text, change the classes, no retraining.

Everything else is standard YOLOv8 — CSPDarknet backbone, PAFPN topology, YOLOv8 bbox head regression, task-aligned assigner for training.

## Tensor shapes you'll see throughout

For a typical forward call with batch size `B`, input image `H×W`, and `N` text prompts (classes):

| Tensor | Shape | Where |
|---|---|---|
| `image` | `(B, 3, H, W)` | input |
| `img_feats` (P3/P4/P5) | `(B, C_l, H/s, W/s)` for `s ∈ {8, 16, 32}` | after backbone/neck |
| `texts` | `List[List[str]]` outer len `B`, inner len `N` | input |
| `txt_feats` | `(B, N, 512)` | after CLIP text encoder, normalized |
| `txt_masks` | `(B, N)` bool | 1 = real class, 0 = padding |
| `cls_logits` (per level) | `(B, N, H/s, W/s)` | from contrastive head |
| `bbox_preds` (per level) | `(B, 4·reg_max, H/s, W/s)` | standard YOLOv8 regression |

Notice: **`N` (number of classes) is dynamic per batch**. That's the whole point.

## The detector — `yolo_world/models/detectors/yolo_world.py`

This is the orchestration layer. It doesn't implement any novel math on its own; it threads the text through the three standard YOLOv8 stages (backbone → neck → head) by overriding the places where image-only flow was assumed.

### Registration and init — `yolo_world.py:11-23`

```python
@MODELS.register_module()
class YOLOWorldDetector(YOLODetector):
    def __init__(self, *args, mm_neck=False, num_train_classes=80,
                 num_test_classes=80, **kwargs):
        self.mm_neck = mm_neck              # use VL neck (text-guided) or plain neck
        self.num_train_classes = num_train_classes
        self.num_test_classes = num_test_classes
        super().__init__(*args, **kwargs)
```

- **`@MODELS.register_module()`** — registers with mmengine so configs can instantiate via `dict(type='YOLOWorldDetector', ...)`.
- **Inherits from mmyolo's `YOLODetector`** — which inherits from mmdet's `BaseDetector`. We get the standard `train_step`/`val_step`/`test_step` machinery for free.
- **`mm_neck`** — boolean flag. `True` means use the vision-language neck (`YOLOWorldPAFPN`, which accepts `txt_feats`). `False` means a plain PAFPN that ignores text. You'll see `mm_neck=True` in every real config — the flag exists mostly to allow ablations.
- **`num_train_classes` / `num_test_classes`** — controls the vocabulary size padding. During training we pad vocab to 80; during testing we read the number from the actual `txt_feats`.

### The three task-mode methods — `yolo_world.py:25-56`

mmdet's `BaseDetector` dispatches to one of three methods depending on the runner's mode:

- **`loss()`** — training (called during `train_step`)
- **`predict()`** — inference with post-processing (called during `val_step`/`test_step`)
- **`_forward()`** — raw forward, no post-processing (used for ONNX export, debugging, etc.)

All three share the same structure:

```python
img_feats, txt_feats, txt_masks = self.extract_feat(batch_inputs, batch_data_samples)
# then hand off to self.bbox_head.<loss|predict|forward>(img_feats, txt_feats, txt_masks, ...)
```

So the *meaningful* code is `extract_feat`. The three task methods are thin wrappers.

There's one interesting detail at **line 47** in `predict()`:

```python
self.bbox_head.num_classes = txt_feats[0].shape[0]
```

The head's `num_classes` attribute is **reset from the actual text feature count** on every predict call. This is how dynamic vocabulary works at inference — the number of classes literally depends on how many strings were in your input.

### The heart of it — `extract_feat()` at `yolo_world.py:75-105`

This is where the dual-path happens:

```python
def extract_feat(self, batch_inputs, batch_data_samples):
    txt_feats = None
    if batch_data_samples is None:
        texts = self.texts                 # path A: reparameterized, texts were pre-baked
        txt_feats = self.text_feats
    elif isinstance(batch_data_samples, dict) and 'texts' in batch_data_samples:
        texts = batch_data_samples['texts']
    elif isinstance(batch_data_samples, list) and hasattr(batch_data_samples[0], 'texts'):
        texts = [data_sample.texts for data_sample in batch_data_samples]
    elif hasattr(self, 'text_feats'):
        texts = self.texts                 # path A again
        txt_feats = self.text_feats
    else:
        raise TypeError('batch_data_samples should be dict or list.')

    if txt_feats is not None:
        # reparameterized / cached path: skip the text encoder
        img_feats = self.backbone.forward_image(batch_inputs)
    else:
        # dual-path: run image AND text through the multimodal backbone
        img_feats, (txt_feats, txt_masks) = self.backbone(batch_inputs, texts)

    if self.with_neck:
        if self.mm_neck:
            img_feats = self.neck(img_feats, txt_feats)   # fusion point 1
        else:
            img_feats = self.neck(img_feats)
    return img_feats, txt_feats, txt_masks
```

Things to notice:

1. **Text can come from three places**: from the batch samples (normal training/inference), from cached `self.text_feats` (after `reparameterize()`), or from `batch_data_samples['texts']` (dict-style API). The tangle of `if/elif` is just routing.

2. **Two code paths downstream of that decision:**
   - **Dual path** — `self.backbone(batch_inputs, texts)` runs *both* image and text. This is what happens during training and normal inference.
   - **Image-only path** — `self.backbone.forward_image(batch_inputs)` skips the text encoder because text was already encoded earlier (via `reparameterize()`). This is the "prompt-then-detect" fast inference mode the README talks about.

3. **The neck receives `txt_feats` when `mm_neck=True`**. This is **fusion point 1** in the diagram above: text modulates the feature pyramid. When `mm_neck=False` the neck is a plain YOLOv8 PAFPN and text only influences the head.

4. **The head always receives `txt_feats`** (it's in the return tuple regardless). So head-level fusion is always on.

### The reparameterization trick — `yolo_world.py:58-61`

```python
def reparameterize(self, texts: List[List[str]]) -> None:
    self.texts = texts
    self.text_feats = self.backbone.forward_text(texts)
```

This is small but conceptually important. It does two things:
1. **Stores the text prompts** on `self` as a fixed vocabulary.
2. **Runs the text encoder once** and caches the resulting embeddings as `self.text_feats`.

From this point on, every `extract_feat` call takes the short path (image-only through the backbone, text features read from the cached tensor). The CLIP text encoder is never touched again for this vocabulary. That's why `reparameterize()` makes inference 2-3× faster — you amortize text encoding across an entire dataset.

The `tools/reparameterize_yoloworld.py` script goes even further: it bakes the cached `self.text_feats` **into the head's weights** (by replacing `ContrastiveHead` with an equivalent `Conv2d`), so the model can be exported to ONNX/TFLite as a vanilla closed-vocab detector. That's a separate topic we can explore later if you want.

---

## The `SimpleYOLOWorldDetector` variant

Below `YOLOWorldDetector` in the same file is `SimpleYOLOWorldDetector` (lines 108-232). It's what `configs/prompt_tuning_coco/` and `configs/image_prompts/` use. Same skeleton, with two differences:

- **`self.embeddings`** is a learnable `nn.Parameter` initialized from a saved `.npy` file. Instead of running the CLIP text encoder every forward pass, the model carries a fixed set of prompt embeddings as parameters that can be trained.
- **Optional `adapter`** — a 2-layer MLP that refines the embeddings. Used for image-prompt adaptation (the CLIP visual features → text-like space trick described in the V2.1 update doc).

Think of it as the "frozen backbone + fine-tuned prompts" mode. Same architecture, different training regime. You can skip it for now — the main `YOLOWorldDetector` is what's doing the heavy lifting.

---

## Ready for Part 2?

Next section will cover the **backbone** — specifically `MultiModalYOLOBackbone`:

- How image and text features are produced in parallel
- How the CLIP text encoder (HuggingFace `CLIPTextModelWithProjection`) is wrapped
- What `forward_image()` / `forward_text()` / `forward()` each do
- How text masks are built (why `txt_masks` exists and what it's used for)
- Why the text encoder is frozen and what that means for gradient flow

Want me to continue straight into Part 2, or do you want to sit with Part 1 first and ask questions?

---

<!-- Section: Part 2 — The backbone -->

# Part 2 — The backbone

The file `yolo_world/models/backbones/mm_backbone.py` (241 lines) defines **four** backbone classes:

| Class | Lines | Role |
|---|---|---|
| `HuggingVisionBackbone` | 14-55 | Image backbone built from a HuggingFace vision model (CLIP ViT, etc.). Alternative to YOLOv8 CSPDarknet. |
| `HuggingCLIPLanguageBackbone` | 58-137 | The real CLIP text encoder wrapper |
| `PseudoLanguageBackbone` | 140-188 | A "fake" text encoder that reads precomputed embeddings from disk |
| **`MultiModalYOLOBackbone`** | **191-240** | **The top-level container that holds an image model + text model side-by-side** |

In the real configs, `MultiModalYOLOBackbone` wraps:
- **image model = YOLOv8 `CSPDarknet`** (from mmyolo, not defined in this file)
- **text model = `HuggingCLIPLanguageBackbone`** (wrapping HuggingFace CLIP)

Let me walk through each of the three that matter. I'll save `PseudoLanguageBackbone` for the end as a footnote — it's a deployment optimization, not a core concept.

---

## 2.1 — `MultiModalYOLOBackbone` — the dual-path container

This is the simplest and the most important. It's literally **two sub-models side by side**, with a thin `forward()` that calls both.

### Init — `mm_backbone.py:191-207`

```python
@MODELS.register_module()
class MultiModalYOLOBackbone(BaseModule):
    def __init__(self, image_model, text_model, frozen_stages=-1,
                 with_text_model=True, init_cfg=None):
        super().__init__(init_cfg)
        self.with_text_model = with_text_model
        self.image_model = MODELS.build(image_model)           # YOLOv8 CSPDarknet
        if self.with_text_model:
            self.text_model = MODELS.build(text_model)         # HuggingCLIPLanguageBackbone
        else:
            self.text_model = None
        self.frozen_stages = frozen_stages
        self._freeze_stages()
```

- **`MODELS.build(image_model)`** — mmengine's registry pattern. The config passes a dict like `dict(type='YOLOv8CSPDarknet', arch='P5', deepen_factor=1.0, ...)` and mmengine instantiates it. Same for `text_model`.
- **`with_text_model=True`** is the default. You set it to `False` only if you're running the reparameterized/deployed mode where text embeddings were already baked in.
- **`frozen_stages`** — how many early stages of the *image* backbone to freeze. Standard mmyolo convention: `-1` means unfrozen, `0` means freeze stem, `2` means freeze stem + first two stages, etc. Used when fine-tuning.

### The three forward methods — `mm_backbone.py:225-240`

```python
def forward(self, image, text):
    img_feats = self.image_model(image)
    if text is not None and self.with_text_model:
        txt_feats = self.text_model(text)
        return img_feats, txt_feats
    else:
        return img_feats, None

def forward_text(self, text):
    assert self.with_text_model, "forward_text() requires a text model"
    txt_feats = self.text_model(text)
    return txt_feats

def forward_image(self, image):
    return self.image_model(image)
```

Three forward methods for three different call sites you saw in Part 1:

| Method | Called from | When |
|---|---|---|
| `forward(image, text)` | `extract_feat()` normal path | Training + first inference pass |
| `forward_image(image)` | `extract_feat()` after reparameterize | Repeated inference (text already cached) |
| `forward_text(texts)` | `reparameterize()` | Building the cache |

That's the entire architecture of the "multi-modal backbone" — **two models, three convenience methods to combine their outputs**. There's no cross-attention between image and text here. Fusion happens later (in the neck and head). At this stage, the two streams are completely independent.

### What shapes come out

**`img_feats`** is a `tuple` of 3 or 4 tensors (the multi-scale feature maps):
```
img_feats[0]: (B, 256, H/8,  W/8)    # P3 — fine detail, small objects
img_feats[1]: (B, 512, H/16, W/16)   # P4 — mid
img_feats[2]: (B, 512, H/32, W/32)   # P5 — coarse, large objects
```

The exact channel counts depend on the model size (S/M/L/X). L-640 has 256/512/512.

**`txt_feats`** is a 3-D tensor (we'll see why in the next subsection):
```
txt_feats: (B, N, 512)   # B = batch size, N = number of text prompts, 512 = CLIP projection dim
```

Note the dual return value — `forward()` returns `(img_feats, txt_feats)` as a tuple. But when you trace it upstream in Part 1, `extract_feat()` does:
```python
img_feats, (txt_feats, txt_masks) = self.backbone(batch_inputs, texts)
```

**Wait — `txt_masks`?** Where does that come from? `MultiModalYOLOBackbone.forward()` only returns two things. The answer is: the `text_model` call inside it (`self.text_model(text)`) returns a *pair* `(txt_feats, txt_masks)` which gets assigned to the local variable `txt_feats`. Then when the caller unpacks it, the outer tuple is actually `(img_feats, (txt_feats, txt_masks))`. Let me show you in the next subsection.

---

## 2.2 — `HuggingCLIPLanguageBackbone` — the CLIP text encoder

This is where text strings become 512-dim embeddings. It's a wrapper around HuggingFace's `CLIPTextModelWithProjection`.

### Init — `mm_backbone.py:58-77`

```python
@MODELS.register_module()
class HuggingCLIPLanguageBackbone(BaseModule):
    def __init__(self, model_name, frozen_modules=(), dropout=0.0,
                 add_mask=False, training_use_cache=False, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.frozen_modules = frozen_modules
        self.training_use_cache = training_use_cache
        self.add_mask = add_mask
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        clip_config = CLIPTextConfig.from_pretrained(model_name, attention_dropout=dropout)
        self.model = CLIPTP.from_pretrained(model_name, config=clip_config)
        self._freeze_modules()
```

- **`model_name`** — something like `'openai/clip-vit-base-patch32'`. HuggingFace does the heavy lifting: downloads weights (~600 MB on first run, cached to `~/.cache/huggingface/`), builds the transformer, loads pretrained state dict.
- **`self.tokenizer`** — the CLIP BPE tokenizer. Converts text strings to input IDs.
- **`self.model = CLIPTP`** — `CLIPTextModelWithProjection`. This is the full CLIP text tower, *including* the learned projection that maps the 512-dim transformer output into CLIP's shared image-text embedding space. That projection is important — without it, you'd get raw transformer outputs that aren't aligned with image features.
- **`frozen_modules`** — list of module name prefixes to freeze. In real configs it's set to `["all"]`, meaning the entire text encoder is frozen. This is critical — we'll come back to why.

### The forward pass — `mm_backbone.py:86-113`

This is the method that actually turns strings into tensors. It's doing a few things at once, so let me annotate it:

```python
def forward(self, text: List[List[str]]) -> Tensor:
    # (1) Validate that every sample in the batch has the same number of prompts
    num_per_batch = [len(t) for t in text]
    assert max(num_per_batch) == min(num_per_batch), (
        'number of sequences not equal in batch')

    # (2) Flatten (B, N) → (B*N,) list of strings for the tokenizer
    text = list(itertools.chain(*text))

    # (3) Optionally build a mask marking which entries are real vs padding
    if self.add_mask:
        text_mask = torch.tensor([x != self.pad_value for x in text],
                                 requires_grad=False).to(self.model.device)

    # (4) Tokenize: list of strings → input_ids + attention_mask
    text = self.tokenizer(text=text, return_tensors='pt', padding=True)
    text = text.to(device=self.model.device)

    # (5) Run the CLIP text tower, under no_grad if the whole encoder is frozen
    if len(self.frozen_modules) > 0:
        with torch.no_grad():
            txt_outputs = self.model(**text)
            txt_feats = txt_outputs.text_embeds
    else:
        txt_outputs = self.model(**text)
        txt_feats = txt_outputs.text_embeds

    # (6) L2-normalize the 512-dim embeddings — this is essential for the contrastive head later
    txt_feats = txt_outputs.text_embeds
    txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)

    # (7) Unflatten back into (B, N, 512)
    txt_feats = txt_feats.reshape(-1, num_per_batch[0], txt_feats.shape[-1])

    # (8) Reshape the mask the same way, or set to None
    if self.add_mask:
        text_mask = text_mask.reshape(-1, num_per_batch[0]).to(txt_feats)
    else:
        text_mask = None

    return txt_feats, text_mask
```

Now the things I want you to internalize:

### Step (1) — the rectangular-batch constraint

Every sample in a batch must have the **same number** of text prompts. If batch item 0 has 80 classes and batch item 1 has 79, the assertion fires. This is why the configs pad the vocabulary to a fixed size during training (`num_train_classes=80`) and why padding entries `[' ']` are added to fill out short prompt lists. It also makes clear why class count is per-batch, not per-sample.

### Step (5) — freezing and `torch.no_grad()`

This is a subtle but important bit. If `frozen_modules` is non-empty (e.g. `["all"]`), the forward is wrapped in `torch.no_grad()`. That means:

1. **No gradient is computed through the text encoder.** All CLIP parameters are effectively non-trainable.
2. **No activations are stored for backprop** — huge memory savings during training.
3. The text encoder is a 60-125M parameter transformer; running it under no_grad makes it behave like a lookup table: strings in, embeddings out, no learning.

**Why freeze?** Two reasons:
- CLIP was pretrained on 400M image-text pairs — it's already very good at mapping language to a visually-meaningful space. Fine-tuning it risks catastrophic forgetting of rare concepts.
- The open-vocabulary property depends on the text embeddings being **in the same space CLIP was originally trained in**. If you fine-tune the text encoder on your detection dataset, embeddings drift toward the classes you trained on, and zero-shot on new classes degrades.

So: CLIP text encoder = frozen lookup. Image backbone = trainable. Everything in between (neck, head) = trainable. This is the core division of labor.

### Step (6) — L2 normalization

```python
txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
```

Every text embedding is projected to the unit sphere. **This is critical for the contrastive head.** Without it, the dot product `img_embed · txt_embed` in the head would reflect both the *direction* (semantic similarity) and the *magnitude* (how confident CLIP was about the word). You only want direction. Normalizing makes `img_embed · txt_embed = cosine_similarity(img_embed, txt_embed)`, which is what a contrastive classifier needs.

You'll see the head does the same L2-normalize to the *image* embeddings, so classification becomes pure cosine similarity.

### Step (8) — what is `text_mask`?

This is a `(B, N)` float tensor where each entry is `1.0` if the prompt was a real class and `0.0` if it was a `' '` padding entry. Remember from the V2.1 update doc: the authors discovered that treating padding as a real class during training helps open-vocab generalization. But they still want to *mask it out of the classification loss* so padding doesn't get gradient signal as a positive.

`add_mask=True` in V2.1 configs, `False` in older V2 configs. The mask flows all the way through the neck and head and is used when computing the classification loss. We'll see it again in Part 4.

---

## 2.3 — `HuggingVisionBackbone` — the alternative image encoder

`mm_backbone.py:14-55`. This is an *alternative* image backbone — it wraps any HuggingFace vision model (e.g. CLIP ViT-L) as the image encoder instead of YOLOv8's CSPDarknet.

```python
def forward(self, image: Tensor) -> Tuple[Tensor]:
    encoded_dict = self.image_model(pixel_values=image, output_hidden_states=True)
    hidden_states = encoded_dict.hidden_states
    img_feats = encoded_dict.get('reshaped_hidden_states', hidden_states)
    img_feats = [img_feats[i] for i in self.image_out_indices]
    return tuple(img_feats)
```

It asks the HF model for its full list of hidden states, then picks specific layers via `image_out_indices` to serve as the P3/P4/P5 equivalents. There are configs in `configs/pretrain/` that use the "clip_large" variant, which uses this backbone to pull multi-scale features from a CLIP ViT-L model.

In practice: **most YOLO-World configs, including the ones we ran, use YOLOv8 CSPDarknet**, not this class. But it's there if you want to experiment with a ViT image backbone. It's a minor variant — I'd skip it for now and focus on the CSPDarknet path.

---

## 2.4 — `PseudoLanguageBackbone` — the deployment shortcut

`mm_backbone.py:140-188`. This class exists to skip the CLIP text encoder entirely at inference. You precompute text embeddings once (with a standalone script like `tools/generate_text_prompts.py`), save them as a pickle file `{"cat": tensor(512), "dog": tensor(512), ...}`, and point this class at the file.

```python
def __init__(self, text_embed_path, test_embed_path=None, init_cfg=None):
    super().__init__(init_cfg)
    self.text_embed = torch.load(text_embed_path, map_location='cpu')   # dict[str] -> (512,)
    ...

def forward_text(self, text):
    ...
    text_embeds = torch.stack(
        [text_embed_dict[x.split("/")[0]] for x in text])
    ...
    return text_embeds
```

Key things:

- **It's a `dict` lookup**, not a neural network. Text goes in, a `(512,)` tensor comes out, no tokenizer, no transformer.
- **`x.split("/")[0]`** is a little hack: in the LVIS class names some contain synonyms like `"cat/feline"` and this takes the first one.
- **You lose the ability to handle arbitrary text at inference** — if a word isn't in the precomputed dict, it crashes. But you save ~600 MB of VRAM and ~50 ms of latency per forward pass.

It's used in configs that need to export to constrained devices (ONNX, TFLite) or in very-high-throughput deployments. For our playing around with `image_demo.py`, we use `HuggingCLIPLanguageBackbone` and don't touch this.

---

## Summary of Part 2

Mental model: **the backbone is two completely parallel towers — YOLOv8 CSPDarknet for images, frozen CLIP for text — glued by a container class that just calls both.** There is zero cross-talk at this stage.

| Question | Answer |
|---|---|
| What goes in? | `image: (B, 3, H, W)` + `texts: List[List[str]]` shape `(B, N)` |
| What comes out? | `img_feats: Tuple[(B, C, H/8, W/8), (B, C, H/16, W/16), (B, C, H/32, W/32)]` + `txt_feats: (B, N, 512)` + `txt_masks: (B, N)` |
| Who's trainable? | CSPDarknet: yes. CLIP text encoder: no (frozen via `torch.no_grad()`) |
| What's the crucial preprocessing? | **L2-normalize `txt_feats`** so the contrastive head computes cosine similarities. |
| What's the crucial constraint? | Every sample in the batch must have **the same number** of prompts. |

---

## Ready for Part 3?

Next up is the **neck** — `yolo_world/models/necks/yolo_world_pafpn.py` plus the supporting layer `MaxSigmoidAttnBlock` in `yolo_world/models/layers/yolo_bricks.py`. This is where the first actual **vision-language fusion** happens: text gates the image feature maps at P3/P4/P5 through a sigmoid-attention mechanism. It's the most interesting piece of the architecture from a "how does fusion actually work" perspective, so it's the piece I'd most like to do justice to.

Want me to continue straight into Part 3, or pause to discuss Part 2 first?

---

<!-- Section: Part 3 — The neck (VL-PAN) -->

# Part 3 — The neck (VL-PAN)

This is the most interesting part of YOLO-World because it's where the novel contribution lives: the idea of using text embeddings to **gate** image features across the detector's feature pyramid.

Two files are involved:

- **`yolo_world/models/necks/yolo_world_pafpn.py`** — the top-level PAN that wires together the standard YOLOv8 multi-scale topology.
- **`yolo_world/models/layers/yolo_bricks.py`** — the building blocks, most importantly `MaxSigmoidAttnBlock` (the novel fusion op) and `MaxSigmoidCSPLayerWithTwoConv` (the CSP block that contains it).

Let me do this in three layers, from outermost to innermost:

1. What a PAFPN is (refresher)
2. How YOLOWorldPAFPN modifies the topology to thread text through
3. The actual fusion math inside `MaxSigmoidAttnBlock`

---

## 3.1 — Refresher: what a PAFPN does

Before we add text, the plain **YOLOv8 PAFPN** (Path Aggregation Feature Pyramid Network) is just this:

```
P5 ──────┐                                    ┌────→ P5'
         ▼        
         upsample +                                   │
P4 ───▶─ concat            (bottom-up path)           ▲
         ▼
        CSPBlock       
         ▼
         upsample + 
P3 ──▶──concat                               ▼     │
         ▼
        CSPBlock ──▶── downsample + concat → CSPBlock ──▶── P3'
                                       │      │
                                       └──────┴──downsample + concat → CSPBlock ──▶── P4'
```

Three feature maps come in (P3, P4, P5 from the backbone, at strides 8, 16, 32). They pass through:
- **A top-down pathway** — info flows from coarse (P5) to fine (P3), giving fine levels semantic context.
- **A bottom-up pathway** — info flows back up, giving coarse levels localization precision.

Between every pair of levels there's a **CSP block** (Cross-Stage Partial) that processes the concatenation. Three scales in → three enriched scales out. That's PAFPN.

YOLOWorldPAFPN's innovation is: **replace the CSP blocks with text-aware CSP blocks**, and pass `txt_feats` through every one. Every fusion point in the pyramid gets told what the user is looking for.

---

## 3.2 — `YOLOWorldPAFPN` — the top-level neck

### Init — `yolo_world_pafpn.py:16-48`

```python
class YOLOWorldPAFPN(YOLOv8PAFPN):
    def __init__(self,
                 in_channels, out_channels,
                 guide_channels,                # <-- 512 (CLIP dim)
                 embed_channels,                # <-- per-level intermediate dim
                 num_heads,                     # <-- per-level attention heads
                 deepen_factor=1.0, widen_factor=1.0,
                 num_csp_blocks=3, ...):
        self.guide_channels = guide_channels
        self.embed_channels = embed_channels
        self.num_heads = num_heads
        self.block_cfg = block_cfg
        super().__init__(in_channels=in_channels, out_channels=out_channels, ...)
```

The key additions over the standard YOLOv8 PAFPN are the three text-related parameters:

- **`guide_channels`** — the text feature dimension. Always 512 for CLIP base. This is the "guide" that will modulate the image features. (The terminology comes from the YOLO-World paper: they call text the *guide* for the image features, which is a little quirky, but that's the naming used in the code.)
- **`embed_channels`** — a per-level list specifying the intermediate channel count inside the attention block at each pyramid level. Usually something like `[128, 256, 512]` for S/M/L variants.
- **`num_heads`** — per-level multi-head attention heads. Usually `[4, 8, 16]`.

### `build_top_down_layer` / `build_bottom_up_layer` — `yolo_world_pafpn.py:50-104`

```python
def build_top_down_layer(self, idx: int) -> nn.Module:
    block_cfg = copy.deepcopy(self.block_cfg)
    block_cfg.update(
        dict(in_channels=..., out_channels=...,
             guide_channels=self.guide_channels,            # <-- 512
             embed_channels=make_round(self.embed_channels[idx - 1], ...),
             num_heads=make_round(self.num_heads[idx - 1], ...),
             num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
             add_identity=False, norm_cfg=..., act_cfg=...))
    return MODELS.build(block_cfg)
```

These are **override hooks** from the parent `YOLOv8PAFPN` class. The parent class does the PAFPN wiring but delegates "what CSP block do I use between levels?" to these two methods. By overriding them, `YOLOWorldPAFPN` swaps in a **`MaxSigmoidCSPLayerWithTwoConv`** (the text-aware variant) everywhere the parent would have used a plain `CSPLayerWithTwoConv`.

This is a nice pattern: the parent class handles the complicated topology (upsample, downsample, concat, ordering), and the child just swaps the block type. Same PAFPN structure, different blocks.

### `forward` — `yolo_world_pafpn.py:106-146`

```python
def forward(self, img_feats: List[Tensor], txt_feats: Tensor = None) -> tuple:
    assert len(img_feats) == len(self.in_channels)

    # 1) reduce layers (unchanged from YOLOv8)
    reduce_outs = []
    for idx in range(len(self.in_channels)):
        reduce_outs.append(self.reduce_layers[idx](img_feats[idx]))

    # 2) top-down path
    inner_outs = [reduce_outs[-1]]
    for idx in range(len(self.in_channels) - 1, 0, -1):
        feat_high = inner_outs[0]
        feat_low = reduce_outs[idx - 1]
        upsample_feat = self.upsample_layers[...](feat_high)
        top_down_layer_inputs = torch.cat([upsample_feat, feat_low], 1)
        inner_out = self.top_down_layers[...](top_down_layer_inputs, txt_feats)  # <-- text goes in
        inner_outs.insert(0, inner_out)

    # 3) bottom-up path
    outs = [inner_outs[0]]
    for idx in range(len(self.in_channels) - 1):
        feat_low = outs[-1]
        feat_high = inner_outs[idx + 1]
        downsample_feat = self.downsample_layers[idx](feat_low)
        out = self.bottom_up_layers[idx](
            torch.cat([downsample_feat, feat_high], 1), txt_feats)            # <-- text goes in
        outs.append(out)

    # 4) out layers
    results = []
    for idx in range(len(self.in_channels)):
        results.append(self.out_layers[idx](outs[idx]))
    return tuple(results)upsample + 
```

If you compare this to the stock `YOLOv8PAFPN.forward()`, the **only** difference is that `self.top_down_layers[...](...)` and `self.bottom_up_layers[...](...)` now accept a second argument: **`txt_feats`**. Everything else — the reduce layers, upsample/downsample, concatenation order — is identical.

So structurally: the neck's topology is pure YOLOv8. The text just rides along as an extra argument passed to every CSP block.

How many fusion points does that give us? For a 3-level pyramid:
- **2 top-down blocks** (P5→P4 fusion, P4→P3 fusion)
- **2 bottom-up blocks** (P3→P4 fusion, P4→P5 fusion)

= **4 text-conditioned fusion points** per forward pass. Four opportunities for text to gate image features.

---

## 3.3 — `MaxSigmoidCSPLayerWithTwoConv` — the text-aware CSP block

`yolo_bricks.py:256-309`. This is what gets instantiated for each of those 4 fusion points.

### Init — `yolo_bricks.py:260-301`

```python
class MaxSigmoidCSPLayerWithTwoConv(CSPLayerWithTwoConv):
    def __init__(self, in_channels, out_channels, guide_channels, embed_channels,
                 num_heads=1, expand_ratio=0.5, num_blocks=1, ...):
        super().__init__(in_channels=in_channels, out_channels=out_channels,
                         expand_ratio=expand_ratio, num_blocks=num_blocks, ...)

        self.final_conv = ConvModule((3 + num_blocks) * self.mid_channels,
                                     out_channels, 1, ...)

        self.attn_block = MaxSigmoidAttnBlock(
            self.mid_channels, self.mid_channels,
            guide_channels=guide_channels,
            embed_channels=embed_channels,
            num_heads=num_heads, ...)
```

It inherits from mmyolo's standard `CSPLayerWithTwoConv` and adds **one** new sub-module: a `MaxSigmoidAttnBlock` stored as `self.attn_block`.

Remember: a CSP block splits its input into two branches, runs one through a stack of bottleneck blocks, and concatenates the results. This layer appends **one extra branch** — the text-guided attention output — to that concatenation.

### Forward — `yolo_bricks.py:303-309`

This is dense but not long:

```python
def forward(self, x: Tensor, guide: Tensor) -> Tensor:
    x_main = self.main_conv(x)                                              # 1x1 project
    x_main = list(x_main.split((self.mid_channels, self.mid_channels), 1))  # split into 2 halves
    x_main.extend(blocks(x_main[-1]) for blocks in self.blocks)             # run bottleneck stack
    x_main.append(self.attn_block(x_main[-1], guide))                       # <-- text-guided branch
    return self.final_conv(torch.cat(x_main, 1))                            # concat all branches
```

Let me trace it with `num_blocks=3` (typical for L-sized models):

1. `self.main_conv(x)` — 1×1 conv that projects input to `2 * mid_channels`.
2. `.split(...)` — split along channel dim into two halves: `x_main = [half_A, half_B]`. This is CSP: half goes through processing, half skips.
3. `x_main.extend(...)` — run `half_B` through the sequential bottleneck blocks. After the loop, `x_main = [half_A, half_B, out_B_1, out_B_2, out_B_3]` (5 tensors).
4. **`x_main.append(self.attn_block(x_main[-1], guide))`** — this is the text fusion line. It takes the *last* bottleneck output (`out_B_3`) and runs it through `MaxSigmoidAttnBlock`, with the text features as the "guide". Appends the result, so `x_main` is now 6 tensors.
5. `torch.cat(x_main, 1)` — concatenate all 6 along the channel dim, then run `final_conv` to project back to `out_channels`.

So the text branch is effectively a **parallel sibling** to the last bottleneck output, not a replacement. Standard image processing + one extra text-modulated feature map, concatenated together.

Note that the text only conditions the **last** bottleneck output, not every one of them. The intuition: run the normal CNN processing stack to build up spatial features, then at the end, let text have a say in *which* of those spatial locations matter.

---

## 3.4 — `MaxSigmoidAttnBlock` — the heart of the fusion

Now we're at the atom. `yolo_bricks.py:15-97`. Everything above was plumbing; this is where the actual vision-language math happens.

### Init — `yolo_bricks.py:19-66`

```python
def __init__(self, in_channels, out_channels, guide_channels, embed_channels,
             kernel_size=3, padding=1, num_heads=1, ..., with_scale=False, ...):
    super().__init__(init_cfg=init_cfg)
    conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

    assert (out_channels % num_heads == 0 and
            embed_channels % num_heads == 0), \
        'out_channels and embed_channels should be divisible by num_heads.'
    self.num_heads = num_heads
    self.head_channels = embed_channels // num_heads
    self.use_einsum = use_einsum

    # (A) 1x1 conv that projects image features to the attention embed space
    self.embed_conv = ConvModule(in_channels, embed_channels, 1, ...) \
                      if embed_channels != in_channels else None

    # (B) Linear that projects text features from CLIP space (512) into embed space
    self.guide_fc = Linear(guide_channels, embed_channels)

    # (C) Per-head learned bias for the attention scores
    self.bias = nn.Parameter(torch.zeros(num_heads))

    # (D) Optional learned scale
    if with_scale:
        self.scale = nn.Parameter(torch.ones(1, num_heads, 1, 1))
    else:
        self.scale = 1.0

    # (E) Main 3x3 conv that processes the image features (will be gated by attention)
    self.project_conv = conv(in_channels, out_channels, kernel_size, ...)
```

Five learnable components:

| Module | What it does | Input dim | Output dim |
|---|---|---|---|
| `embed_conv` | Project image features into the attention embed space | `in_channels` (image feat C) | `embed_channels` (e.g. 256) |
| `guide_fc` | Project text features into the **same** embed space | `guide_channels` (512) | `embed_channels` (256) |
| `bias` | Per-head learned offset for attention scores | — | `(num_heads,)` |
| `scale` | Optional per-head learned scale | — | `(1, num_heads, 1, 1)` or `1.0` |
| `project_conv` | Main 3×3 conv on the image features (the thing being gated) | `in_channels` | `out_channels` |

**The crucial design point:** `embed_conv` and `guide_fc` both produce tensors in the same `embed_channels`-dimensional space. That's the **shared attention space** where image features and text embeddings can be directly compared.

### Forward — `yolo_bricks.py:68-97`

OK, this is the money shot. Let me annotate carefully with shapes — assuming batch size `B`, `C_in` input channels, `H×W` spatial, `N` text prompts, `M` heads, `d = embed_channels / M` per-head dim.

```python
def forward(self, x: Tensor, guide: Tensor) -> Tensor:
    B, _, H, W = x.shape
    #  x    : image features      (B, C_in, H, W)
    #  guide: text features       (B, N, 512)

    # --- 1. Project text into the attention space, reshape into heads ---
    guide = self.guide_fc(guide)                                  # (B, N, embed_channels)
    guide = guide.reshape(B, -1, self.num_heads, self.head_channels)
    #  guide: (B, N, M, d)

    # --- 2. Project image features into the same attention space, reshape into heads ---
    embed = self.embed_conv(x) if self.embed_conv is not None else x     # (B, embed_channels, H, W)
    embed = embed.reshape(B, self.num_heads, self.head_channels, H, W)
    #  embed: (B, M, d, H, W)

    # --- 3. Compute attention weights: image at each spatial location vs every text token ---
    if self.use_einsum:
        attn_weight = torch.einsum('bmchw,bnmc->bmhwn', embed, guide)
    else:
        # equivalent manual matmul (for ONNX export where einsum is flaky)
        batch, m, channel, height, width = embed.shape
        _, n, _, _ = guide.shape
        embed = embed.permute(0, 1, 3, 4, 2)           # (B, M, H, W, d)
        embed = embed.reshape(batch, m, -1, channel)    # (B, M, H*W, d)
        guide = guide.permute(0, 2, 3, 1)              # (B, M, d, N)
        attn_weight = torch.matmul(embed, guide)        # (B, M, H*W, N)
        attn_weight = attn_weight.reshape(batch, m, height, width, n)
    #  attn_weight: (B, M, H, W, N)
    #  — at every spatial location (h,w), for every head m,
    #    we have a similarity score against every text prompt n.

    # --- 4. MAX over text prompts — this is "MaxSigmoid" ---
    attn_weight = attn_weight.max(dim=-1)[0]            # (B, M, H, W)
    #  Each pixel now has a single score per head: "how strongly does this pixel
    #  match the BEST-matching text prompt in the vocabulary?"

    # --- 5. Scale and bias (standard attention polish) ---
    attn_weight = attn_weight / (self.head_channels**0.5)               # (B, M, H, W)
    attn_weight = attn_weight + self.bias[None, :, None, None]

    # --- 6. Sigmoid into a gate in [0, 1], optional learned scale ---
    attn_weight = attn_weight.sigmoid() * self.scale     # (B, M, H, W)

    # --- 7. Run the main conv on image features, reshape into heads, and GATE ---
    x = self.project_conv(x)                             # (B, out_channels, H, W)
    x = x.reshape(B, self.num_heads, -1, H, W)           # (B, M, out_channels/M, H, W)
    x = x * attn_weight.unsqueeze(2)                     # element-wise gate across heads
    #         attn_weight: (B, M, 1, H, W) after unsqueeze
    x = x.reshape(B, -1, H, W)                           # (B, out_channels, H, W)
    return x
```

Let me walk through what this mechanism **is doing conceptually**, because the einsum is deceptive:

### Step 3 in plain English

For each spatial location `(h, w)` on the feature map, for each attention head `m`, the block computes:

> *"What is the dot product between my image embedding at this pixel and the text embedding of prompt n, for every n?"*

This gives a `(B, M, H, W, N)` tensor — a similarity map between every pixel and every word.

### Step 4 — the "Max" in MaxSigmoid

This is the novel bit. Instead of doing softmax over text (which would force a probability distribution across prompts), the block does **max over text**:

```python
attn_weight = attn_weight.max(dim=-1)[0]   # collapse the N dim by taking the max
```

The result is a `(B, M, H, W)` map where each pixel asks:

> *"Does this pixel match **any** of the text prompts well?"*

It's an OR-like aggregation: if pixel `(h, w)` strongly matches `"power socket"` OR `"person"` OR any other prompt, it gets a high score. The identity of which prompt won doesn't matter here — that's decided later in the head. The neck just wants a **text-aware saliency map**.

Why max and not softmax?
- **Max is symmetric over vocabulary size.** With softmax, if you have 80 prompts versus 800, the softmaxed scores would be distributed very differently even for the same actual visual content. Max doesn't care.
- **Max preserves signal strength.** If one prompt is a perfect match, the max keeps that raw score; softmax would normalize it down.
- **It's a gate, not a classification.** The neck isn't trying to decide the class here — that's the head's job. It just wants to say "this region is interesting for at least one text prompt we care about".

### Steps 5-6 — turn scores into gates

Divide by `sqrt(d)` (standard attention scaling to prevent saturated softmax/sigmoid), add a learned bias, then sigmoid:

```python
attn_weight = attn_weight.sigmoid() * self.scale   # in [0, 1]
```

Now `attn_weight` is an `(B, M, H, W)` gate map in `[0, 1]`.

### Step 7 — multiplicative gating

```python
x = self.project_conv(x)                   # normal 3x3 conv on image features
x = x.reshape(B, self.num_heads, -1, H, W) # split channel dim into heads
x = x * attn_weight.unsqueeze(2)           # gate by sigmoid attention
x = x.reshape(B, -1, H, W)                 # collapse heads back
```

The gate is applied element-wise to the output of a standard 3×3 conv. Regions of the feature map that matched some text prompt get amplified (gate ≈ 1); regions that matched nothing get suppressed (gate ≈ 0).

**This is the entire fusion mechanism.** Notice what it's *not*:

- **Not cross-attention** in the transformer sense (no softmax, no weighted sum of values).
- **Not late fusion** (it's applied inside the neck, not only at the head).
- **Not symmetric** — image features are modulated by text, but text is not modulated by image. The flow is one-way: text → image.
- **Very cheap.** One linear on text, one 1×1 conv on image, an einsum, a max, a sigmoid, one 3×3 conv, and element-wise multiply. This is why YOLO-World can still run in real time.

### Why call it an "attention block"?

The name is a bit loose. A transformer attention block has **query / key / value** with softmax and a value aggregation. This block only has **query-key similarity** (image embed × text embed) and uses the result as a gate on the image features directly. There's no value aggregation. It's more accurately a **"text-conditioned gating"** layer or **"feature-wise modulation"** — a close cousin of FiLM (Feature-wise Linear Modulation), but using sigmoid gating instead of scale+shift.

The authors called it attention because the einsum math looks like attention and because the paper leans into the "vision-language attention" framing. Under the hood, it's simpler and cheaper than full cross-attention.

---

## 3.5 — Other blocks in `yolo_bricks.py` (skim)

The file has several variants you don't need to read in detail:

| Block | `yolo_bricks.py` lines | Purpose |
|---|---|---|
| **`MaxSigmoidAttnBlock`** | 15-97 | **Default** — what we just walked through |
| `RepMatrixMaxSigmoidAttnBlock` | 100-174 | Reparameterization target — the `guide_fc` is collapsed into a `guide_weight` parameter for ONNX export |
| `RepConvMaxSigmoidAttnBlock` | 177-253 | Another reparameterization target using per-head 1×1 convs |
| `MaxSigmoidCSPLayerWithTwoConv` | 256-309 | **Default** CSP block — what the PAFPN uses |
| `RepMaxSigmoidCSPLayerWithTwoConv` / `RepConvMaxSigmoidCSPLayerWithTwoConv` | 312-423 | Reparameterized CSP block variants |
| `ImagePoolingAttentionModule` | 426-502 | Used only by `YOLOWorldDualPAFPN` — enriches text features using pooled image info (reverse direction fusion) |
| `VanillaSigmoidBlock` | 505-547 | Ablation baseline — a plain conv with no text gating |
| `EfficientCSPLayerWithTwoConv` | 550-601 | CSP block using the vanilla sigmoid block instead of MaxSigmoid |

You can ignore the `Rep*` variants until you get to the reparameterization tour. They're functionally equivalent to `MaxSigmoidAttnBlock` at inference but structured so `reparameterize_yoloworld.py` can fold text embeddings directly into their weights (turning `guide_fc @ txt_feats` into a fixed `guide_weight` parameter). Same math, different parameter layout.

---

## 3.6 — `YOLOWorldDualPAFPN` — the richer variant

`yolo_world_pafpn.py:149-235`. Same architecture as `YOLOWorldPAFPN` plus **one more fusion step in the reverse direction**.

Between the top-down and bottom-up paths, it inserts:

```python
txt_feats = self.text_enhancer(txt_feats, inner_outs)
```

where `text_enhancer` is `ImagePoolingAttentionModule`. It pools the image feature maps with adaptive max pool to a 3×3 grid per level, then does a **real transformer-style attention** (softmax, not max-sigmoid) with text features as query and pooled image features as key/value, and adds the result back to `txt_feats`.

So the dual PAFPN runs:
1. Top-down with text-gated attention on image features (text → image)
2. **Enhance text features with pooled image context** (image → text)
3. Bottom-up with text-gated attention using the *enhanced* text

This is the only place in the architecture where information flows from image → text. In practice, the vanilla `YOLOWorldPAFPN` (no dual) is what the L/M/S configs use. DualPAFPN is used in a few experimental configs and gives ~0.5-1 AP bump at the cost of extra FLOPs.

---

## Summary of Part 3

**Mental model:** The neck is a standard YOLOv8 PAFPN where every CSP block gets an extra text argument. Each of the 4 CSP blocks in the pyramid contains a `MaxSigmoidAttnBlock` that:

1. Projects image features and text features into a shared attention space.
2. Computes a similarity map: *every pixel vs every text prompt*.
3. **Max-reduces** over text prompts to get a single "interestingness" score per pixel.
4. Sigmoids the score into a `[0, 1]` gate.
5. Uses the gate to **multiplicatively modulate** a main 3×3 conv output.

The gate says *"pay attention to this region because it could be one of the things the user is looking for"*. It does not say which one — that's decided in the head.

| Question | Answer |
|---|---|
| Where does text flow into image features? | 4 points: 2 top-down CSP blocks + 2 bottom-up CSP blocks |
| What's the attention math? | `max_n(img_embed_hw · text_embed_n)` → sigmoid → gate |
| Why max instead of softmax? | Vocabulary-size invariant, preserves signal strength, only need saliency not classification |
| Is it real cross-attention? | No — no value aggregation. It's a text-conditioned gate. Cousin of FiLM. |
| How expensive? | Cheap. One 1×1 conv + linear + einsum + sigmoid per fusion point. Keeps real-time. |
| Does text get updated by image? | No in `YOLOWorldPAFPN`. Yes in `YOLOWorldDualPAFPN` (via `ImagePoolingAttentionModule`). |

---

## Ready for Part 4?

Part 4 is the **head** — `yolo_world/models/dense_heads/yolo_world_head.py`. That's where:
- The contrastive classification logits get computed (`ContrastiveHead` / `BNContrastiveHead`)
- Text features meet image features *again* for the actual class prediction
- The bbox regression still happens via standard YOLOv8 (this part is boring)
- The text mask from Part 2 finally gets used (in the loss function)
- You'll see how the training loss works and why padding is handled specially

It's the second half of vision-language fusion and completes the story. Same drill — want me to continue, or pause for questions on Part 3?

---

<!-- Section: Multi-head combination (follow-up to Part 3) -->

Great catch — this is a subtle point and the answer is a bit unusual. The short version: **the heads are never explicitly combined inside `MaxSigmoidAttnBlock`.** Each head gates a different slice of the output channels, and the downstream convolution layer does the cross-head mixing implicitly.

Let me trace exactly how that works, because it's different from standard transformer multi-head attention.

## What each head produces

Look at `yolo_bricks.py:68-91` again, tracking the per-head dimension carefully:

```python
guide = self.guide_fc(guide)                                   # (B, N, embed_channels)
guide = guide.reshape(B, -1, self.num_heads, self.head_channels)
#  guide: (B, N, M, d)   ← every head gets its own slice of the text embedding

embed = self.embed_conv(x)                                     # (B, embed_channels, H, W)
embed = embed.reshape(B, self.num_heads, self.head_channels, H, W)
#  embed: (B, M, d, H, W)  ← every head gets its own slice of the image embedding

attn_weight = torch.einsum('bmchw,bnmc->bmhwn', embed, guide)
#  (B, M, H, W, N)  ← similarity per-head, per-pixel, per-text-prompt

attn_weight = attn_weight.max(dim=-1)[0]                       # (B, M, H, W)
attn_weight = attn_weight / (self.head_channels**0.5)
attn_weight = attn_weight + self.bias[None, :, None, None]     # (M,) bias, one per head
attn_weight = attn_weight.sigmoid() * self.scale               # (B, M, H, W), in [0, 1]
```

The `reshape` into heads **partitions** both `embed_channels` and the CLIP dim into `M` non-overlapping chunks of `d = embed_channels / M` each. The einsum `'bmchw,bnmc->bmhwn'` then contracts over the per-head channel dimension `c`, so **each head `m` only ever sees its own slice** of the image embedding and its own slice of the text embedding. Head 0 computes similarity in subspace 0, head 1 in subspace 1, and so on. They never talk to each other during the similarity computation.

After the max-over-prompts and sigmoid, you get `attn_weight: (B, M, H, W)` — **one saliency map per head**. Each head can have a completely different opinion about which pixels are interesting.

## The gating step — channel-wise, not merged

Now the key lines (`yolo_bricks.py:93-96`):

```python
x = self.project_conv(x)                   # (B, out_channels, H, W)
x = x.reshape(B, self.num_heads, -1, H, W) # (B, M, out_channels/M, H, W)
x = x * attn_weight.unsqueeze(2)           # broadcast: (B, M, 1, H, W)
x = x.reshape(B, -1, H, W)                 # (B, out_channels, H, W)
```

Read this carefully:

1. `project_conv` produces `out_channels` feature maps via a standard 3×3 conv. These channels are a flat block of `out_channels` feature planes.
2. `reshape(B, M, -1, H, W)` **implicitly partitions** those `out_channels` feature planes into `M` groups, each holding `out_channels / M` channels. Head 0 "owns" channels `[0 : out_channels/M]`, head 1 owns `[out_channels/M : 2*out_channels/M]`, etc.
3. `attn_weight.unsqueeze(2)` gives a `(B, M, 1, H, W)` tensor. The `1` broadcasts across the "channels I own" dimension — so head `m`'s saliency map `attn_weight[:, m, :, :]` multiplies **every channel in head `m`'s group**.
4. The final `reshape(B, -1, H, W)` is just a view — it flattens the `(M, out_channels/M)` pair back into `out_channels`. No arithmetic, just a memory reinterpretation.

So when a single head "fires" at pixel `(h, w)`, it amplifies its own slice of channels at that pixel and leaves every other head's slice untouched. The gated output is a **channel-wise concatenation** of `M` independently-gated feature chunks — not an average, not a sum, not a projection.

Visually:

```
output channels (out_channels total):

┌──────────────┬──────────────┬──────────────┬──────────────┐
│  head 0      │  head 1      │  head 2      │  head 3      │
│  gated by    │  gated by    │  gated by    │  gated by    │
│  sal_map[0]  │  sal_map[1]  │  sal_map[2]  │  sal_map[3]  │
└──────────────┴──────────────┴──────────────┴──────────────┘
  out_channels/M     /M             /M              /M
```

Each block lives in its own channel slice. If head 2's saliency is near-zero at pixel `(h, w)`, channels in its block are suppressed at that pixel while the other heads' blocks pass through unchanged.

## So where does cross-head mixing happen?

**Downstream, in the next convolution.** The attention block returns a `(B, out_channels, H, W)` tensor. That tensor then goes back to the enclosing `MaxSigmoidCSPLayerWithTwoConv`, which does (from `yolo_bricks.py:303-309`):

```python
def forward(self, x: Tensor, guide: Tensor) -> Tensor:
    x_main = self.main_conv(x)
    x_main = list(x_main.split((self.mid_channels, self.mid_channels), 1))
    x_main.extend(blocks(x_main[-1]) for blocks in self.blocks)
    x_main.append(self.attn_block(x_main[-1], guide))    # ← returns gated-per-head tensor
    return self.final_conv(torch.cat(x_main, 1))          # ← CROSS-HEAD MIXING HAPPENS HERE
```

`self.final_conv` is a plain 1×1 `ConvModule` that takes the concatenated stack of branches (including the attention block's output) and projects it down to `out_channels`. **A 1×1 conv is a per-pixel linear combination across every input channel.** Since each head's gated channels are sitting side-by-side in the channel dim, that 1×1 conv freely mixes them — a single output channel is a learned weighted sum that pulls from head 0's region, head 1's region, etc.

So the combination is:

1. **Per-head gating inside `MaxSigmoidAttnBlock`** → each head gates its own channel slice.
2. **Per-pixel linear mixing in `final_conv`** of the enclosing CSP block → mixes across heads by treating them as adjacent channel groups.
3. **Further mixing in downstream conv layers** (the next CSP block, the head's stem convs, etc.) → every subsequent convolution can learn to combine the per-head information however it wants.

## How this differs from standard multi-head attention

Standard transformer MHA does it like this:

```
per head: Q_m K_m^T softmax → A_m V_m = head_output_m    # (d,)
concat all heads along channel: [head_out_0, ..., head_out_M-1]  # (M*d,)
apply W_o linear to mix across heads: W_o @ concat → final         # (d_model,)
```

The `W_o` projection is the explicit cross-head mixing. It's part of the attention block itself.

YOLO-World's `MaxSigmoidAttnBlock` **does not have a `W_o` equivalent inside the block**. It does the channel partition and gating, then trusts the downstream `final_conv` (and all subsequent convolutions) to do the mixing. You can think of `final_conv` as effectively playing the role of `W_o` for the attention block — it's the first operation that mixes across head slices.

This is a deliberate simplification. A proper `W_o` would cost another `embed_channels × embed_channels` matmul per fusion point. By piggybacking on the CSP block's `final_conv` that was going to run anyway, they save that compute without losing the cross-head mixing capability.

## Why have multiple heads at all, then?

Because each head learns a different **subspace** of the embed space in which to compare image and text. With `num_heads=16` on the L model:

- The `guide_fc` linear learns to project 512-d CLIP embeddings into 16 different 32-d subspaces.
- The `embed_conv` 1×1 learns to project image features into the *same* 16 subspaces.
- Different subspaces can specialize — one head might become attuned to shape similarity, another to texture, another to color, another to size context, and so on. Nothing forces this specialization, but SGD tends to discover it because the per-head biases and the downstream mixing give each head room to develop its own personality.
- Each head's per-pixel sigmoid gate is a **different "opinion"** about whether a pixel is text-relevant, based on what that subspace is sensitive to.

Because the gates multiply disjoint channel slices, **the disagreements are preserved**: if head 0 says yes and head 3 says no, head 0's channels at that pixel stay hot while head 3's channels go cold. The next conv then decides how to integrate those signals.

It's a cheaper, more convolution-friendly analog of the usual multi-head design — heads as *specialized filters* rather than as *parallel attention streams that get unified before leaving the block*.

## Worth noting: the bias is also per-head

`self.bias = nn.Parameter(torch.zeros(num_heads))` — a single scalar per head that offsets its saliency scores before the sigmoid. This lets each head learn its own default "permissiveness":

- A head with large negative bias is off by default unless its query-key similarity really spikes.
- A head with large positive bias is on by default unless its similarity really drops.

So each head gets **both** its own subspace and its own baseline. The combination gives them a lot of flexibility to carve up the gating work differently.

---

**Summary answer to your question:** Yes, saliency is computed per head (one `(H, W)` map per head). The heads are *not* explicitly combined inside `MaxSigmoidAttnBlock`. Instead, the block's output channels are partitioned `M`-ways and each head gates its own slice. The cross-head mixing is deferred to the very next layer — `final_conv` in the enclosing `MaxSigmoidCSPLayerWithTwoConv`, which is a 1×1 conv that freely recombines channels across head boundaries. This is a cheaper alternative to having a `W_o` projection inside the attention block itself.

Want me to continue into Part 4 (the head), or any more questions on the neck first?

---

<!-- Section: Part 4 — The head -->

# Part 4 — The head

The head is split into two layers of classes that mirror the mmyolo pattern:

- **`YOLOWorldHead`** (lines 347-749) — the *outer* class. It's a thin wrapper that plugs into mmdet's dispatch (loss/predict/forward) and delegates the actual computation to a head module. It owns the loss function.
- **`YOLOWorldHeadModule`** (lines 134-289) — the *inner* class. Does the actual work: the convolutional stem, bbox regression, classification via contrastive head.

In between these, three small "contrastive head" classes do the image–text dot product for classification:

- **`ContrastiveHead`** (27-64) — L2-normalized cosine-similarity classifier
- **`BNContrastiveHead`** (67-106) — BatchNorm variant (used in V2 and V2.1)
- **`RepBNContrastiveHead`** (109-131) — reparameterized variant (plain `Conv2d`, no text input)

And one alternative head module used for reparameterization:

- **`RepYOLOWorldHeadModule`** (292-344) — subclass of `YOLOWorldHeadModule` that uses `RepBNContrastiveHead`.

Let me walk through them bottom-up: contrastive head → head module → outer head with loss.

---

## 4.1 — `ContrastiveHead` — the cosine-similarity classifier

`yolo_world_head.py:27-64`. This is the smallest but the most conceptually important class in the file. It's ~30 lines that replace YOLOv8's `Conv2d(channels, num_classes, 1)` classifier with a **parameter-free dot-product**.

### Init

```python
@MODELS.register_module()
class ContrastiveHead(BaseModule):
    def __init__(self, embed_dims, init_cfg=None, use_einsum=True):
        super().__init__(init_cfg=init_cfg)
        self.bias = nn.Parameter(torch.zeros([]))                        # scalar
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07)) # ≈ 2.66, scalar
        self.use_einsum = use_einsum
```

**Notice what's missing.** No `num_classes` in the constructor. No `Conv2d(..., num_classes)`. Nothing that depends on how many classes you have. That's the whole point — the "classifier" here is just a dot product against whatever text embeddings you hand it at forward time.

Only two learnable scalars:

- **`bias`** — a single scalar added to the logits after the dot product. One scalar for the entire classifier.
- **`logit_scale`** — a single scalar *temperature* controlling how sharp the distribution gets. Initialized to `log(1/0.07) ≈ 2.66`, which comes directly from OpenAI CLIP — that's the temperature value CLIP was trained with. So at init, YOLO-World's classifier is effectively running with CLIP's exact sharpness, and SGD is free to adjust from there.

### Forward

```python
def forward(self, x: Tensor, w: Tensor) -> Tensor:
    x = F.normalize(x, dim=1, p=2)     # x: (B, C, H, W) → L2-norm along channel dim
    w = F.normalize(w, dim=-1, p=2)    # w: (B, N, C)   → L2-norm along feature dim

    if self.use_einsum:
        x = torch.einsum('bchw,bkc->bkhw', x, w)
    else:
        # equivalent matmul reshape, as before
        ...

    x = x * self.logit_scale.exp() + self.bias
    return x
```

Three steps:

**(1) L2-normalize both sides.**

- `x` is the per-pixel image embedding. Shape `(B, C, H, W)` with `C = embed_dims` (usually 512 for YOLO-World). After normalizing along `dim=1`, every pixel is a unit vector in `C`-dim space.
- `w` is the text features. Shape `(B, N, C)` with `N = num_prompts`. After normalizing along the last dim, every text embedding is a unit vector.

Remember from Part 2: `txt_feats` was *already* L2-normalized by `HuggingCLIPLanguageBackbone.forward()`. So the `F.normalize(w)` here is slightly redundant in practice, but important for correctness — it ensures the invariant holds even if someone feeds in unnormalized text features.

**(2) The dot product.**

```python
x = torch.einsum('bchw,bkc->bkhw', x, w)
```

This is the atom of open-vocabulary classification. Let's parse the einsum:
- `bchw` — image features: batch, channels, height, width
- `bkc` — text features: batch, class index `k`, channels
- `bkhw` — output: batch, class index, height, width

Contraction is over `c`. At every pixel `(h, w)`, for every class `k`, we compute:

> `logit[b, k, h, w] = sum_c (x[b, c, h, w] * w[b, k, c])`

Since both `x[b, :, h, w]` and `w[b, k, :]` are unit vectors, the dot product equals **cosine similarity**. So after this line, each pixel has a cosine-similarity score against each text prompt. That's the open-vocabulary classification logit.

**(3) Scale and bias.**

```python
x = x * self.logit_scale.exp() + self.bias
```

Cosine similarity lives in `[-1, 1]`, which is way too narrow for a logit — after sigmoid it would collapse most scores to around 0.5. The `logit_scale.exp()` multiplier *stretches* the range so the downstream sigmoid has something to work with. With the initial `logit_scale = log(1/0.07) ≈ 2.66`, `exp(2.66) ≈ 14.3`, so a cosine of 0.9 becomes a logit of ~12.9, and sigmoid of that is ~0.9999. A cosine of 0.5 becomes a logit of ~7.2, sigmoid ~0.999. A cosine of 0 stays at 0, sigmoid 0.5.

`self.bias` is a single learned scalar that shifts all logits uniformly. You can think of it as a global "how much do we trust the model on anything at all" offset.

### What this buys you

- **Dynamic vocabulary.** `num_classes` is never baked in anywhere; it's whatever the second dim of `w` happens to be. Change the prompts, change the classes, with no weight changes.
- **Zero-shot generalization for free.** Classification is cosine similarity in CLIP space. If CLIP's text encoder knows a concept like "IEC power inlet", the contrastive head immediately supports detecting it even though YOLO-World was never trained on that word — the semantic distance from known training concepts just has to be manageable.
- **Only two scalars to train.** Because the classifier has effectively zero parameters that depend on `num_classes`, the number of trainable classification parameters is literally just `bias` and `logit_scale`. All the "where" information lives in `cls_preds` (the convolutional stem that produces `x`). All the "what" information lives in `txt_feats` (the frozen CLIP embeddings). The head just binds them together.

### Why `logit_scale.exp()` and not just a raw parameter?

Keeping the scale as `exp(param)` guarantees it's always positive — cosine similarity that gets negated by a negative multiplier would flip the sign of the classifier. By parameterizing in log-space, SGD can freely increase or decrease the scale without risking a sign flip. Same trick Anthropic/OpenAI used in CLIP itself.

---

## 4.2 — `BNContrastiveHead` — the batch-norm variant

`yolo_world_head.py:67-106`. Same idea, **one key difference**: replaces `F.normalize(x)` with a BatchNorm.

```python
def __init__(self, embed_dims, norm_cfg, init_cfg=None, use_einsum=True):
    super().__init__(init_cfg=init_cfg)
    self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
    self.bias = nn.Parameter(torch.zeros([]))
    self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))   # ← init is -1.0, not log(1/0.07)
    self.use_einsum = use_einsum

def forward(self, x: Tensor, w: Tensor) -> Tensor:
    x = self.norm(x)                     # BatchNorm2d on image features
    w = F.normalize(w, dim=-1, p=2)      # still L2-norm on text features
    # same einsum & scale/bias as ContrastiveHead
    ...
```

**Why?** The answer is buried in the *reparameterization story*, which is the whole reason BN exists in the head.

The plain `ContrastiveHead` uses `F.normalize(x)`. L2 normalization is **not a linear operation** — each pixel's norm depends on that pixel's own values, so you can't represent it as a matrix multiply. This is a problem when you want to bake the text embeddings into a static `Conv2d` for deployment: you can fold a linear operation into a conv, but you can't fold an L2 norm.

BatchNorm, by contrast, is **affine at inference time**: after training, every BN layer is just `(x - running_mean) / sqrt(running_var + eps) * weight + bias`, which is a linear per-channel transformation. You can *absorb* a BN layer into the preceding or following conv by folding its scale and shift into the conv's weights.

So:
- **Training / normal inference**: `BNContrastiveHead` trains with BN + text-embedding dot product.
- **Deployment / reparameterization**: `tools/reparameterize_yoloworld.py` folds the BN + text-embedding matmul into a fixed `Conv2d(embed_dims, num_classes, 1)`. The text encoder gets dropped entirely, the head becomes a vanilla closed-vocab classifier, and the model exports cleanly to ONNX/TFLite.

That's why the `logit_scale` initialization is different here: `-1.0` (i.e. the multiplier is `exp(-1) ≈ 0.37`) instead of `log(1/0.07) ≈ 2.66`. BN's output isn't constrained to the unit sphere, so the logits have a much larger natural range and don't need as aggressive a stretch factor. The comment at line 84 says `# use -1.0 is more stable` — empirical choice during training.

**Which head is in use for which configs?** In the V2.1 configs (including the `l_stage1` weights you downloaded), `use_bn_head=True`. In older V1 configs, it was `use_bn_head=False`. The L-640 and L-1280 models we ran both use `BNContrastiveHead`.

---

## 4.3 — `RepBNContrastiveHead` — the reparameterized static variant

`yolo_world_head.py:109-131`. This is what `BNContrastiveHead` collapses into after reparameterization.

```python
class RepBNContrastiveHead(BaseModule):
    def __init__(self, embed_dims, num_guide_embeds, norm_cfg, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        self.conv = nn.Conv2d(embed_dims, num_guide_embeds, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        x = self.conv(x)
        return x
```

No text features anywhere. It's a `BatchNorm` followed by a `Conv2d(embed_dims, num_classes, 1)`. The `Conv2d`'s weights are just the reshaped text embeddings from the reparameterization step, baked in as static parameters. `num_guide_embeds` is the fixed vocabulary size.

When the reparameterize script runs, it takes the `logit_scale.exp() * (BN(x) @ txt_feats.T)` math from `BNContrastiveHead` and rewrites it as:
```
BN(x) @ (logit_scale.exp() * txt_feats.T)
```
which is exactly a `Conv2d(embed_dims, N, 1)` whose weights are `(logit_scale.exp() * txt_feats.T)`. The bias becomes the conv's bias term. And the result is a plain `Conv2d` call that ONNX can trivially export.

You won't touch this class unless you're exporting — but it's useful to know it exists because it explains why `BNContrastiveHead` has BN at all.

---

## 4.4 — `YOLOWorldHeadModule` — the actual head

`yolo_world_head.py:134-289`. This inherits from mmyolo's `YOLOv8HeadModule` and is where all the head convolutions live.

### `_init_layers` — `yolo_world_head.py:167-228`

This builds three parallel `ModuleList`s, one entry per pyramid level (P3, P4, P5):

```python
def _init_layers(self) -> None:
    self.cls_preds = nn.ModuleList()          # image → embed_dims feature map
    self.reg_preds = nn.ModuleList()          # image → 4 * reg_max bbox distribution
    self.cls_contrasts = nn.ModuleList()      # (embed, text) → logits

    reg_out_channels = max((16, self.in_channels[0] // 4, self.reg_max * 4))
    cls_out_channels = max(self.in_channels[0], self.num_classes)

    for i in range(self.num_levels):
        # Regression branch: 2x (3x3 conv + BN + SiLU) + 1x1 conv → (4 * reg_max) channels
        self.reg_preds.append(nn.Sequential(
            ConvModule(in_channels=self.in_channels[i], out_channels=reg_out_channels, kernel_size=3, ...),
            ConvModule(in_channels=reg_out_channels,    out_channels=reg_out_channels, kernel_size=3, ...),
            nn.Conv2d(in_channels=reg_out_channels, out_channels=4 * self.reg_max, kernel_size=1)))

        # Classification stem: 2x (3x3 conv + BN + SiLU) + 1x1 conv → embed_dims channels
        self.cls_preds.append(nn.Sequential(
            ConvModule(in_channels=self.in_channels[i], out_channels=cls_out_channels, kernel_size=3, ...),
            ConvModule(in_channels=cls_out_channels,    out_channels=cls_out_channels, kernel_size=3, ...),
            nn.Conv2d(in_channels=cls_out_channels, out_channels=self.embed_dims, kernel_size=1)))

        # Contrastive head for this level
        if self.use_bn_head:
            self.cls_contrasts.append(BNContrastiveHead(self.embed_dims, self.norm_cfg, ...))
        else:
            self.cls_contrasts.append(ContrastiveHead(self.embed_dims, ...))

    proj = torch.arange(self.reg_max, dtype=torch.float)
    self.register_buffer('proj', proj, persistent=False)
```

This is YOLOv8's standard decoupled head — separate classification and regression towers — with one crucial substitution:

**Where stock YOLOv8 does `Conv2d(cls_out_channels, num_classes, 1)` at the tail of the cls branch, YOLO-World does `Conv2d(cls_out_channels, embed_dims, 1)` and then a `ContrastiveHead` outside the sequential stack.**

So `cls_preds[i]` produces a `(B, embed_dims, H, W)` tensor — a per-pixel "query embedding". It's not yet logits; it's a vector that wants to be compared against text. `cls_contrasts[i]` then does that comparison against `txt_feats`.

The regression branch is completely unchanged from YOLOv8: same DFL-style distribution-of-offsets prediction, same `reg_max` quantization, same everything. Bbox regression is text-agnostic.

### `init_weights` — `yolo_world_head.py:155-165`

```python
def init_weights(self, prior_prob=0.01):
    super().init_weights()
    for cls_pred, cls_contrast, stride in zip(self.cls_preds, self.cls_contrasts, self.featmap_strides):
        cls_pred[-1].bias.data[:] = 0.0   # reset bias of final cls_pred 1x1 conv
        if hasattr(cls_contrast, 'bias'):
            nn.init.constant_(
                cls_contrast.bias.data,
                math.log(5 / self.num_classes / (640 / stride)**2))
```

This is small but clever. The last line initializes the contrastive head's `bias` scalar to a value that gives roughly the right prior probability at init. The formula approximates: "if we expect ~5 positive anchors per image across all classes and scales, set the initial bias so that sigmoid(scale * 0 + bias) matches that prior". This is a standard focal-loss init trick from RetinaNet, adapted for the contrastive head. Without it, the model spends the first epochs unlearning a massively wrong prior.

### `forward` — `yolo_world_head.py:246-254`

```python
def forward(self, img_feats, txt_feats, txt_masks):
    assert len(img_feats) == self.num_levels
    txt_feats = [txt_feats for _ in range(self.num_levels)]
    txt_masks = [txt_masks for _ in range(self.num_levels)]
    return multi_apply(self.forward_single, img_feats, txt_feats, txt_masks,
                       self.cls_preds, self.reg_preds, self.cls_contrasts)
```

`multi_apply` from mmdet is a utility that `zip`s the arguments and calls `forward_single` on each tuple, then transposes the results. It's equivalent to:

```python
outs = [self.forward_single(img_feats[i], txt_feats, txt_masks,
                            self.cls_preds[i], self.reg_preds[i], self.cls_contrasts[i])
        for i in range(self.num_levels)]
cls_logits, bbox_preds, [bbox_dist_preds] = zip(*outs)
```

Important detail: **the same `txt_feats` tensor is reused across all 3 levels**. Text doesn't change with scale — the same CLIP embeddings are compared against P3, P4, and P5 feature maps. What changes is the image side.

### `forward_single` — `yolo_world_head.py:256-289`

This is where per-level classification and regression happen:

```python
def forward_single(self, img_feat, txt_feat, txt_masks,
                   cls_pred, reg_pred, cls_contrast):
    b, _, h, w = img_feat.shape

    # --- Classification branch ---
    cls_embed = cls_pred(img_feat)                              # (B, embed_dims, H, W)
    cls_logit = cls_contrast(cls_embed, txt_feat)               # (B, N, H, W)

    # --- Mask out padding classes ---
    if txt_masks is not None:
        txt_masks = txt_masks.view(b, -1, 1, 1).expand(-1, -1, h, w)
        if self.training:
            cls_logit = cls_logit * txt_masks                   # zero out padding logits
            cls_logit[txt_masks == 0] = -10e6                   # then kill them
        else:
            cls_logit[txt_masks == 0] = -10e6                   # kill them at inference too

    # --- Regression branch: predict a discrete distribution over offsets ---
    bbox_dist_preds = reg_pred(img_feat)                        # (B, 4*reg_max, H, W)
    if self.reg_max > 1:
        bbox_dist_preds = bbox_dist_preds.reshape([-1, 4, self.reg_max, h * w]).permute(0, 3, 1, 2)
        bbox_preds = bbox_dist_preds.softmax(3).matmul(self.proj.view([-1, 1])).squeeze(-1)
        bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
    else:
        bbox_preds = bbox_dist_preds

    if self.training:
        return cls_logit, bbox_preds, bbox_dist_preds
    else:
        return cls_logit, bbox_preds
```

Two branches computed in parallel:

#### Classification branch (`cls_embed`, `cls_logit`, mask)

1. `cls_pred(img_feat)` — the 2× conv + 1×1 stem. Produces `(B, embed_dims, H, W)` — a per-pixel query embedding.
2. `cls_contrast(cls_embed, txt_feat)` — the contrastive head we walked through in 4.1. Returns `(B, N, H, W)` of per-pixel, per-class logits.
3. **The text mask is applied.** Remember from Part 2 that `txt_masks` is `(B, N)` with 1 for real classes and 0 for `' '` padding. Here it gets broadcast to `(B, N, H, W)` so it can be applied to the per-pixel logits.
   - At **training time** there's a double-application: first `cls_logit *= txt_masks` (sets padding logits to 0), then `cls_logit[txt_masks == 0] = -10e6`. The `* txt_masks` is there to make the gradient flow correct — multiplying by 0 keeps the gradient structure for positive-weighted BCE, whereas just overwriting with `-10e6` would zero out gradient through the padding anchors. Then the `-10e6` assignment ensures that at inference (or when the loss uses the raw logits), padding prompts can never win.
   - At **inference time**, just `cls_logit[txt_masks == 0] = -10e6` → sigmoid → ~0. Padding prompts are suppressed.
   - **Note:** This mask-based suppression is the V2.1 behavior. In V2, the padding prompts were allowed to score freely and treated as a normal class (see the V2.1 update blog for the motivation). That's why the V2.1 `add_mask=True` path exists in the backbone — so the head can do this suppression.

#### Regression branch (bbox distribution → point prediction)

This part is pure YOLOv8 and has nothing to do with text. Worth understanding though:

1. `reg_pred(img_feat)` produces `(B, 4 * reg_max, H, W)`. For `reg_max=16` (standard), that's 64 channels per pixel.
2. These 64 channels decode as **4 histograms**, one for each side of the box (l, t, r, b offsets from the anchor point), each with `reg_max=16` bins.
3. The softmax + dot product with `self.proj = [0, 1, 2, ..., reg_max-1]` computes the *expected value* of each side's distribution — i.e., the weighted average of the bin indices, which is a soft estimate of the distance in stride units.

This is **Distribution Focal Loss** (DFL) from Generalized Focal Loss: instead of regressing a single scalar per side, you predict a discrete distribution and take its expectation. It gives better gradients and calibrated uncertainty. The `bbox_dist_preds` are returned separately during training so the DFL loss can be computed; at inference only the decoded `bbox_preds` are needed.

The final output `bbox_preds` has shape `(B, 4, H, W)` where the 4 channels are the offsets (l, t, r, b) in stride units from each pixel position. The post-processing later converts these to absolute coordinates using the per-pixel priors.

---

## 4.5 — `YOLOWorldHead` — the outer wrapper

`yolo_world_head.py:347-749`. This is the outer class that plugs into mmdet's dispatch. It inherits from mmyolo's `YOLOv8Head`, overrides `forward`/`loss`/`predict` to accept text, and owns the loss function.

### `forward` / `predict` — `yolo_world_head.py:396-417`

These are trivial delegates:

```python
def forward(self, img_feats, txt_feats, txt_masks):
    return self.head_module(img_feats, txt_feats, txt_masks)

def predict(self, img_feats, txt_feats, txt_masks, batch_data_samples, rescale=False):
    batch_img_metas = [ds.metainfo for ds in batch_data_samples]
    outs = self(img_feats, txt_feats, txt_masks)   # calls forward above
    predictions = self.predict_by_feat(*outs, batch_img_metas=batch_img_metas, rescale=rescale)
    return predictions
```

`self.head_module` is the `YOLOWorldHeadModule` we just walked through. `predict_by_feat` is YOLOv8's standard post-processing: flatten multi-scale outputs, decode bboxes against priors, sigmoid the cls_logits, filter by score threshold, NMS, rescale to original image size. This is also largely unchanged from stock YOLOv8 and I won't dig into it unless you want to.

### `loss_by_feat` — the training loss

This is the long method at lines 428-577. Let me hit the interesting parts.

#### Getting priors (anchor points)

```python
if current_featmap_sizes != self.featmap_sizes_train:
    mlvl_priors_with_stride = self.prior_generator.grid_priors(
        self.featmap_sizes_train, dtype=..., device=..., with_stride=True)
    self.flatten_priors_train = torch.cat(mlvl_priors_with_stride, dim=0)
    self.stride_tensor = self.flatten_priors_train[..., [2]]
```

YOLOv8 is **anchor-free**: each pixel is a single "prior point" at the center of its grid cell, tagged with the stride of that level. `grid_priors` builds a `(total_pixels, 3)` tensor where each row is `[px, py, stride]`. This gets cached across training iterations if the feature map sizes don't change.

#### Flatten predictions across levels

```python
flatten_cls_preds = [cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.num_classes)
                     for cls_pred in cls_scores]
flatten_pred_bboxes = [bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
                       for bbox_pred in bbox_preds]
flatten_pred_dists = [bbox_pred_org.reshape(num_imgs, -1, self.head_module.reg_max * 4)
                      for bbox_pred_org in bbox_dist_preds]

flatten_cls_preds  = torch.cat(flatten_cls_preds, dim=1)    # (B, total_pixels, N)
flatten_pred_bboxes = torch.cat(flatten_pred_bboxes, dim=1)  # (B, total_pixels, 4)
flatten_dist_preds  = torch.cat(flatten_pred_dists, dim=1)   # (B, total_pixels, reg_max*4)
flatten_pred_bboxes = self.bbox_coder.decode(
    self.flatten_priors_train[..., :2], flatten_pred_bboxes, self.stride_tensor[..., 0])
```

All three levels are flattened and stacked along the pixel dimension, so downstream we treat all `total_pixels = H3*W3 + H4*W4 + H5*W5` priors as one big list. The bbox decoder converts the per-pixel `(l, t, r, b)` stride offsets into absolute `(x1, y1, x2, y2)` coordinates using the priors.

#### Label assignment — task-aligned assigner

```python
assigned_result = self.assigner(
    (flatten_pred_bboxes.detach()).type(gt_bboxes.dtype),
    flatten_cls_preds.detach().sigmoid(),
    self.flatten_priors_train,
    gt_labels, gt_bboxes, pad_bbox_flag)

assigned_bboxes = assigned_result['assigned_bboxes']
assigned_scores = assigned_result['assigned_scores']
fg_mask_pre_prior = assigned_result['fg_mask_pre_prior']
```

This is where **training targets are decided**. YOLO-World uses the *Task-Aligned Assigner* from TOOD (the same one YOLOv8 uses), which lives in `yolo_world/assigner/task_aligned_assigner.py`. The core idea:

- For each ground-truth box, compute an **alignment score** for every prior: `alignment = cls_score^α * iou^β`. This is a joint measure of "does this prior confidently classify as the GT class" and "does this prior's predicted box overlap the GT well".
- Pick the top-k priors per GT as positives.
- Positives get targets: `assigned_bboxes` = the GT box, `assigned_scores` = the soft label (alignment score, not a hard 0/1).
- Everything else is a negative (`fg_mask_pre_prior == 0`).

The crucial detail: `assigned_scores` is a **soft label** equal to the alignment score, not a 0/1 indicator. So positive priors aren't told "class X is 1, everything else is 0" — they're told "class X is 0.82 (or whatever the alignment was)". This is important because it couples the classification loss with regression quality: a positive prior with poor IoU gets a lower classification target, which nudges the model to correlate confidence with spatial accuracy.

#### Classification loss — **this is where `txt_masks` is used**

```python
if batch_text_masks is not None:
    cls_weight = batch_text_masks.view(num_imgs, 1, -1).expand(
        -1, flatten_cls_preds.shape[1], -1).to(flatten_cls_preds)

    loss_cls = self.loss_cls(flatten_cls_preds, assigned_scores)
    _loss_cls = (loss_cls * cls_weight).sum(dim=-1)
    loss_cls = _loss_cls.sum()
else:
    loss_cls = self.loss_cls(flatten_cls_preds, assigned_scores).sum()

loss_cls /= assigned_scores_sum
```

Here's the finale for why `txt_masks` exists:

- `self.loss_cls` is typically a **per-element** BCE (binary cross-entropy) with `reduction='none'`, returning a loss tensor of shape `(B, total_pixels, N)`.
- `cls_weight` is the text mask broadcast to the same shape `(B, total_pixels, N)` — 1 for real classes, 0 for padding.
- Multiplying elementwise **zeroes out the loss contribution from padding classes**. So the model isn't penalized for incorrectly scoring padding prompts.

This is the V2.1 improvement from `docs/update_20250123.md`: padding is still in the vocabulary (so the attention/neck gating still works with a fixed 80-class training vocab), but padding is masked *out of the classification loss* so gradients only flow through the real classes. Cleaner signal, ~0.5-1 AP bump on LVIS over V2.

Normalization: `loss_cls /= assigned_scores_sum`. `assigned_scores_sum` is the total mass of positive soft labels across the batch. Dividing by this effectively normalizes by "number of positive samples" but softly — a batch with many high-quality positives gets a larger divisor than a batch with few.

#### Bbox and DFL losses — `yolo_world_head.py:532-570`

```python
assigned_bboxes /= self.stride_tensor       # to stride units
flatten_pred_bboxes /= self.stride_tensor

num_pos = fg_mask_pre_prior.sum()
if num_pos > 0:
    prior_bbox_mask = fg_mask_pre_prior.unsqueeze(-1).repeat([1, 1, 4])
    pred_bboxes_pos = torch.masked_select(flatten_pred_bboxes, prior_bbox_mask).reshape([-1, 4])
    assigned_bboxes_pos = torch.masked_select(assigned_bboxes, prior_bbox_mask).reshape([-1, 4])
    bbox_weight = torch.masked_select(assigned_scores.sum(-1), fg_mask_pre_prior).unsqueeze(-1)

    loss_bbox = self.loss_bbox(pred_bboxes_pos, assigned_bboxes_pos,
                               weight=bbox_weight) / assigned_scores_sum

    # DFL loss on the distribution of box offsets
    pred_dist_pos = flatten_dist_preds[fg_mask_pre_prior]
    assigned_ltrb = self.bbox_coder.encode(
        self.flatten_priors_train[..., :2] / self.stride_tensor,
        assigned_bboxes,
        max_dis=self.head_module.reg_max - 1, eps=0.01)
    assigned_ltrb_pos = torch.masked_select(assigned_ltrb, prior_bbox_mask).reshape([-1, 4])
    loss_dfl = self.loss_dfl(pred_dist_pos.reshape(-1, self.head_module.reg_max),
                             assigned_ltrb_pos.reshape(-1),
                             weight=bbox_weight.expand(-1, 4).reshape(-1),
                             avg_factor=assigned_scores_sum)
```

Two losses here, both **only applied to positive anchors** (via `fg_mask_pre_prior`):

1. **`loss_bbox`** — typically CIoU or GIoU loss between predicted and target boxes. Weighted by the assigner's alignment score so high-quality positives contribute more.
2. **`loss_dfl`** — Distribution Focal Loss on the per-side offset distributions. Encourages the predicted discrete distributions to concentrate around the true `(l, t, r, b)` values in stride units.

Both losses are text-independent. **Text only influences the classification loss, never the regression loss.** The assigner uses text-based class scores to pick positives, but once picked, the bbox math is identical to stock YOLOv8.

#### Return

```python
return dict(loss_cls=loss_cls * num_imgs * world_size,
            loss_bbox=loss_bbox * num_imgs * world_size,
            loss_dfl=loss_dfl * num_imgs * world_size)
```

Three loss terms, all scaled by `num_imgs * world_size`. That scaling normalizes across the world (all GPUs in distributed training) so the effective batch size looks consistent regardless of how many GPUs you have.

mmengine will sum these three into the final loss, weight them by whatever `loss_weight` factors are in the config (typical: `loss_cls=0.5`, `loss_bbox=7.5`, `loss_dfl=1.5`), and call `backward()` on the sum.

---

## 4.6 — `RepYOLOWorldHeadModule` — the reparameterized head module

`yolo_world_head.py:292-344`. This is the head-module counterpart to `RepBNContrastiveHead`. Same structure as `YOLOWorldHeadModule` but uses `RepBNContrastiveHead` instead of `BNContrastiveHead`, and its `forward_single` doesn't take `txt_feat`:

```python
def forward_single(self, img_feat, cls_pred, reg_pred, cls_contrast):
    b, _, h, w = img_feat.shape
    cls_embed = cls_pred(img_feat)
    cls_logit = cls_contrast(cls_embed)       # no text input
    ...
```

`cls_contrast(cls_embed)` — single argument. The text embeddings have been folded into the `Conv2d`'s weights. You only use this after running `tools/reparameterize_yoloworld.py`, which replaces `BNContrastiveHead` with `RepBNContrastiveHead` and packs your fixed vocabulary's CLIP embeddings into the new conv's weights.

At that point your YOLO-World model behaves exactly like a standard YOLOv8 detector with a fixed set of classes — same inputs, same outputs, same ONNX-friendly op graph — but zero-shot generalization is frozen at whatever prompts you baked in.

---

## Summary of Part 4

**Mental model:** The head is YOLOv8's standard decoupled head, but with one surgical modification: `Conv2d(cls, num_classes, 1)` is replaced with `Conv2d(cls, embed_dims, 1) + ContrastiveHead`. Classification logits come from cosine similarity between per-pixel image embeddings and CLIP text embeddings, not from a static learned classifier. The bbox regression branch is completely untouched — YOLO-World is text-aware for *what* but stock YOLOv8 for *where*.

| Question | Answer |
|---|---|
| Where do image and text meet at the head? | `ContrastiveHead.forward(x, w)`: `logit_scale.exp() * cosine(x, w) + bias` |
| How does `num_classes` stay dynamic? | `ContrastiveHead` has no `num_classes` parameter; class count is whatever `w.shape[1]` is at call time |
| Why are there two contrastive head variants? | `ContrastiveHead` uses L2 norm (non-linear, can't be baked). `BNContrastiveHead` uses BatchNorm (linear at inference, can be baked into a `Conv2d` for deployment). V2.1 uses the BN one. |
| What does `txt_masks` do in the head? | Zeroes out padding classes in the loss (training) and sets their logits to `-inf` (inference). New in V2.1. |
| Where does text affect bbox regression? | It doesn't. Regression is pure YOLOv8 DFL. Text only touches classification. |
| Who picks training positives? | `TaskAlignedAssigner` — uses `cls_score^α * iou^β` alignment. Positives get soft labels = alignment score. |
| What are the three loss terms? | `loss_cls` (BCE on contrastive logits, masked by text), `loss_bbox` (CIoU, positives only), `loss_dfl` (DFL on offset distributions, positives only). All weighted by task-aligned scores. |

---

## Where that leaves us

That's the full architecture walkthrough:

1. **Detector** (`yolo_world.py`) — orchestrates the three stages and handles the reparameterize-cache fast path.
2. **Backbone** (`mm_backbone.py`) — two parallel towers: YOLOv8 CSPDarknet + frozen CLIP. No cross-talk.
3. **Neck** (`yolo_world_pafpn.py` + `yolo_bricks.py`) — text gates image features 4× via `MaxSigmoidAttnBlock` inside each CSP block. Multi-head saliency with per-head channel partitioning.
4. **Head** (`yolo_world_head.py`) — contrastive classifier against text embeddings; YOLOv8 DFL bbox regression; task-aligned assignment with text mask on the loss.

## Natural follow-ups if you want more depth

There are a few places we could still dig into:

- **The config file** — reading `configs/pretrain/yolo_world_v2_l_vlpan_bn_*.py` top to bottom to see every piece get wired: model dict, pipeline, dataset combos (Objects365 + GoldG + CC3M), optimizer, schedule.
- **The task-aligned assigner** — `yolo_world/assigner/task_aligned_assigner.py`. Nice standalone piece of math that's worth understanding because it's what shapes every "positive anchor" decision.
- **The reparameterization script** — `tools/reparameterize_yoloworld.py`. We've mentioned it three times now. It's the clearest illustration of why `BNContrastiveHead` uses BN instead of L2-norm, and it's only ~200 lines.
- **The training data pipeline** — `yolo_world/datasets/`. How do Objects365 (detection) and Flickr/GoldG (grounding) get combined into a single batch? This is an interesting engineering piece that's not about the model at all.

You can also of course just run more experiments with your own images and tune prompts — now that you have the mental model, you can reason about *why* a given prompt succeeds or fails.

Any of those interest you, or are you happy to pause here?

---


<!-- Section: Appendix A — CSPDarknet Backbone (detailed) -->

# Appendix A — CSPDarknet Backbone (detailed)

The image backbone used by YOLO-World is **not** defined in the YOLO-World
repo itself — it's pulled in from mmyolo. This appendix drills into the
YOLOv8 CSPDarknet structure to answer:

- What does CSPDarknet look like internally?
- Where are the P3 / P4 / P5 outputs drawn from?
- At what stride does each output come out at?
- How do the model size variants (S / M / L / X) scale?

## Source files

| File | What's in it |
|---|---|
| `third_party/mmyolo/mmyolo/models/backbones/csp_darknet.py:158-299` | `YOLOv8CSPDarknet` class |
| `third_party/mmyolo/mmyolo/models/backbones/base_backbone.py:16-226` | `BaseBackbone`, whose `forward()` assembles stem + stages |
| `third_party/mmyolo/mmyolo/models/layers/yolo_bricks.py:1441-1510` | `CSPLayerWithTwoConv` (used inside each stage) |
| `third_party/mmyolo/mmyolo/models/layers/yolo_bricks.py:34-121` | `SPPFBottleneck` (used in the deepest stage) |
| `third_party/mmyolo/configs/yolov8/yolov8_l_syncbn_fast_8xb16-500e_coco.py` | L-model configuration (`deepen_factor=1.0, widen_factor=1.0, last_stage_out_channels=512`) |

## 1 stem + 4 stages

The backbone has **1 stem and 4 stages**, each applying a stride-2 downsample
conv at its start. After the stem (stride 2) and 4 stages (each stride 2),
the cumulative strides are 2, 4, 8, 16, 32. Each component is indexed 0..4
in the `nn.ModuleList` that `BaseBackbone` iterates over:

```
Input image: (B, 3, H, W)
                    │
                    ▼
┌─────────────────────────────────────┐
│  Index 0: Stem                      │  3×3 Conv, stride=2, BN, SiLU
│  3 → 64 channels                    │  output: (B, 64, H/2, W/2)        stride 2
└─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────┐
│  Index 1: Stage 1                   │  3×3 Conv s=2 (downsample)
│  64 → 128 channels                  │  + CSPLayerWithTwoConv (3 blocks)
│  3 bottleneck blocks                │  output: (B, 128, H/4, W/4)       stride 4
└─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────┐
│  Index 2: Stage 2                   │  3×3 Conv s=2 (downsample)
│  128 → 256 channels                 │  + CSPLayerWithTwoConv (6 blocks)
│  6 bottleneck blocks                │  output: (B, 256, H/8, W/8)       stride 8   ← P3
└─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────┐
│  Index 3: Stage 3                   │  3×3 Conv s=2 (downsample)
│  256 → 512 channels                 │  + CSPLayerWithTwoConv (6 blocks)
│  6 bottleneck blocks                │  output: (B, 512, H/16, W/16)     stride 16  ← P4
└─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────┐
│  Index 4: Stage 4                   │  3×3 Conv s=2 (downsample)
│  512 → last_stage_out_channels      │  + CSPLayerWithTwoConv (3 blocks)
│  3 bottleneck blocks + SPPFBottleneck│ output: (B, 512, H/32, W/32)    stride 32  ← P5
└─────────────────────────────────────┘
```

The architecture is driven by this table in `csp_darknet.py:206-209`:

```python
arch_settings = {
    'P5': [[64, 128, 3, True, False],     # stage 1: in=64,  out=128, 3 blocks, no SPPF
           [128, 256, 6, True, False],    # stage 2: in=128, out=256, 6 blocks, no SPPF
           [256, 512, 6, True, False],    # stage 3: in=256, out=512, 6 blocks, no SPPF
           [512, None, 3, True, True]],   # stage 4: in=512, out=last_stage_out_channels,
                                          #          3 blocks, WITH SPPF
}
```

Each row: `[in_channels, out_channels, num_blocks, add_identity, use_spp]`.

`None` for stage 4's `out_channels` means it uses the config parameter
`last_stage_out_channels`, which is **512** for the L model.

## Where do P3, P4, P5 come from?

The backbone's `out_indices` parameter controls which stages emit outputs.
For YOLOv8 / YOLO-World, the default is:

```python
out_indices=(2, 3, 4)
```

The `forward()` in `base_backbone.py:216-225` iterates through the 5
components (stem + 4 stages) and collects outputs at the requested indices:

| Index | Component | Output channels (L) | Cumulative stride | Returned as |
|---|---|---|---|---|
| 0 | Stem | 64 | 2 | (not returned) |
| 1 | Stage 1 | 128 | 4 | (not returned) |
| **2** | **Stage 2** | **256** | **8** | **`img_feats[0]` → P3** |
| **3** | **Stage 3** | **512** | **16** | **`img_feats[1]` → P4** |
| **4** | **Stage 4** | **512** | **32** | **`img_feats[2]` → P5** |

So P3, P4, P5 are **already at strides 8, 16, 32** when they leave the
backbone. Each stage's leading 3×3 stride-2 conv does the spatial
downsampling. By the time features reach the neck, they're at the right
resolution.

For a 640×640 input image (L-640 model), the feature map sizes are:

| Level | Stride | Spatial size | Channels | Pixels |
|---|---|---|---|---|
| P3 | 8 | 80 × 80 | 256 | 6,400 |
| P4 | 16 | 40 × 40 | 512 | 1,600 |
| P5 | 32 | 20 × 20 | 512 | 400 |
| **Total** | | | | **8,400 anchor points** |

That 8,400 is the number you'd see if you counted `flatten_priors_train` in
the head's loss function — every one of those pixels is a potential
detection anchor.

## Each stage's internal structure

Every stage has two components chained in a `nn.Sequential`:

**1. A downsample conv** — `ConvModule(in_ch, out_ch, kernel=3, stride=2, padding=1)`
   with BN + SiLU. Halves spatial dims, changes channel count.

**2. A `CSPLayerWithTwoConv`** — Cross-Stage Partial block (shape-preserving
   spatially, so stride-1):

```
input (out_ch)
    │
    ▼
main_conv (1×1 stride=1, out_ch → 2 × mid_ch)
    │
    ├─── split ────┐
    │              │
  half_A         half_B
  (skip)           │
                   ├── DarknetBottleneck ──► out_1
                   ├── DarknetBottleneck ──► out_2
                   ├── ...                   ...
                   └── DarknetBottleneck ──► out_N
    │              │         │                │
    └──── cat ─────┴─────────┴────────────────┘
                   │
                final_conv (1×1) → output (out_ch)
```

Where `mid_ch = out_ch * expand_ratio`, and the default `expand_ratio=0.5`,
so `mid_ch = out_ch / 2`.

Each `DarknetBottleneck` is two 3×3 convs with a residual connection
(`x + Conv(Conv(x))`). Stride-1, no channel change — they refine features
in place.

**Stage 4 additionally has `SPPFBottleneck`** (Spatial Pyramid Pooling -
Fast) at the end. It applies max-pooling at kernel size 5 three times
progressively:

```
input → pool_1(input) → pool_2(pool_1) → pool_3(pool_2)
concat(input, pool_1, pool_2, pool_3) → 1×1 conv → output
```

This gives the deepest feature level a larger effective receptive field —
P5 can "see" a bigger area of the input image, which helps detect large
objects.

## Scaling across model sizes

The architecture stays the same for S/M/L/X — only `deepen_factor` and
`widen_factor` change:

| Model | deepen_factor | widen_factor | last_stage_out_channels | P3 ch | P4 ch | P5 ch | Stage 2 blocks | Stage 3 blocks |
|---|---|---|---|---|---|---|---|---|
| S | 0.33 | 0.50 | 1024 | 128 | 256 | 512 | 2 | 2 |
| M | 0.67 | 0.75 | 768 | 192 | 384 | 576* | 4 | 4 |
| L | 1.00 | 1.00 | 512 | 256 | 512 | 512 | 6 | 6 |
| X | 1.00 | 1.25 | 512 | 320 | 640 | 640* | 6 | 6 |

*After `make_divisible(..., 8)` rounding (channels always multiples of 8).

`widen_factor` scales channels (via `make_divisible`), `deepen_factor`
scales block counts (via `make_round`, never below 1). Same topology, just
wider and/or deeper.

## Relation to the YOLO-World pretraining story

A natural question is: is the CSPDarknet backbone trainable during YOLO-World
pretraining, or is it initialized from YOLOv8's COCO-pretrained weights and
frozen?

**During pretraining:** No config ships with `frozen_stages` set for the
image backbone. CSPDarknet is **fully trainable**. The paper doesn't ablate
"pretrain with a frozen YOLOv8 backbone" — it would be an unusual choice,
because pretraining is where detection-specific features are built and
aligned with CLIP's text embedding space.

**During fine-tuning / prompt-tuning / segmentation adaptation:**
`frozen_stages=4` (fully frozen backbone) **is** used in:

```
configs/prompt_tuning_coco/        → frozen_stages=4
configs/image_prompts/             → frozen_stages=4
configs/segmentation/              → frozen_stages=4  # "frozen the image backbone"
```

So the intended workflow is:

1. **Pretrain** with a fully trainable CSPDarknet → learn detection features
   aligned to CLIP space.
2. **Fine-tune / prompt-tune** with a frozen backbone → adapt only the
   neck / head / prompts to a specific domain.

`docs/finetuning.md:76-84` also notes that even if you strip away the
text-guidance machinery entirely (via the `EfficientCSPLayerWithTwoConv`
block that removes the text gating), the pretrained CSPDarknet + neck weights
yield a **stronger detector than stock YOLOv8 on COCO** — because YOLO-World
is pretrained on a much larger and more diverse corpus (Objects365 + GoldG +
CC3M). The text guidance acted as a training-time regularizer that shaped
the features, even if it's removed at deployment.

---

<!-- Section: Appendix B — VL-PAN Tensor Shape Walkthrough -->

# Appendix B — VL-PAN Tensor Shape Walkthrough

This appendix supplements Part 3 with a precise per-step tensor-shape trace
through `YOLOWorldPAFPN.forward()`, for the L-640 model. It also corrects
two mistakes in Part 3:

1. The `main_conv` inside `CSPLayerWithTwoConv` is a **1×1 stride-1 conv**,
   not a 3×3 stride-2 conv. (It's a channel projection, not a downsampler.)
2. `mid_channels = out_channels * expand_ratio = out_channels * 0.5`.
   For a block with `out_channels=512`, `mid_channels = 256` (not 128).

## Component inventory

The neck inherits layer-building from a four-level class hierarchy:

```
YOLOWorldPAFPN → YOLOv8PAFPN → YOLOv5PAFPN → BaseYOLONeck
```

The assembly happens in `BaseYOLONeck.__init__`
(`base_yolo_neck.py:155-175`). It builds **six** `ModuleList`s, but only
four of them actually do work for YOLO-World. The other two are no-ops
inherited from the parent structure:

| ModuleList | Count | What it is in YOLO-World | Source |
|---|---|---|---|
| `reduce_layers` | 3 | **`nn.Identity()`** — no-op | `YOLOv8PAFPN.build_reduce_layer` |
| `upsample_layers` | 2 | **`nn.Upsample(scale_factor=2, mode='nearest')`** | `YOLOv5PAFPN.build_upsample_layer` |
| `top_down_layers` | 2 | **`MaxSigmoidCSPLayerWithTwoConv`** — text-aware | `YOLOWorldPAFPN.build_top_down_layer` |
| `downsample_layers` | 2 | **3×3 stride-2 `ConvModule`** (keeps channels) | `YOLOv5PAFPN.build_downsample_layer` |
| `bottom_up_layers` | 2 | **`MaxSigmoidCSPLayerWithTwoConv`** — text-aware | `YOLOWorldPAFPN.build_bottom_up_layer` |
| `out_layers` | 3 | **`nn.Identity()`** — no-op | `YOLOv5PAFPN.build_out_layer` |

So the `reduce_layers` and `out_layers` are hooks-for-subclasses that
YOLO-World (and YOLOv8) leaves as identity. The actual text-image fusion
happens only in the 4 CSP blocks (2 top-down + 2 bottom-up).

## Channel configuration (L-640)

From the YOLOv8-L base config + YOLO-World neck overrides:

```python
# From YOLOv8-L base
in_channels  = [256, 512, 512]   # P3, P4, P5 channel counts (L, widen=1.0, last_stage=512)
out_channels = [256, 512, 512]   # same — PAFPN preserves per-level channel counts

# From the YOLO-World config
neck_embed_channels = [128, 256, 256]   # embed_channels per level for the attention blocks
neck_num_heads      = [4,   8,   8]     # heads per level (head_channels = 32 for all)
text_channels       = 512               # guide_channels = CLIP-B/32 dim
```

## Input shapes

For a 640×640 input image, the backbone hands the neck this tuple:

```
img_feats[0] = P3:  (B, 256, 80, 80)   stride 8
img_feats[1] = P4:  (B, 512, 40, 40)   stride 16
img_feats[2] = P5:  (B, 512, 20, 20)   stride 32

txt_feats:          (B, N, 512)        N text prompts, 512-dim CLIP embeddings
```

## Complete forward trace

`YOLOWorldPAFPN.forward()` at `yolo_world_pafpn.py:106-146`, step by step.
I'll label intermediate tensors `T#` so you can track them.

### Stage 0 — Reduce (no-op)

```python
reduce_outs = [self.reduce_layers[idx](img_feats[idx]) for idx in range(3)]
```

Since all reduce layers are `Identity`:

```
reduce_outs[0] = P3:  (B, 256, 80, 80)    ← unchanged
reduce_outs[1] = P4:  (B, 512, 40, 40)    ← unchanged
reduce_outs[2] = P5:  (B, 512, 20, 20)    ← unchanged
```

### Stage 1 — Top-down path

Info flows **coarse → fine** (P5 down to P3). The loop runs for `idx = 2, 1`.

```python
inner_outs = [reduce_outs[-1]]   # start with P5
for idx in range(len(self.in_channels) - 1, 0, -1):   # idx: 2, 1
    feat_high = inner_outs[0]
    feat_low = reduce_outs[idx - 1]
    upsample_feat = self.upsample_layers[len(self.in_channels) - 1 - idx](feat_high)
    top_down_layer_inputs = torch.cat([upsample_feat, feat_low], 1)
    inner_out = self.top_down_layers[len(self.in_channels) - 1 - idx](
        top_down_layer_inputs, txt_feats)
    inner_outs.insert(0, inner_out)
```

**Iteration 1 — `idx=2`** (fuse P5 with P4):

| Step | Computation | Shape |
|---|---|---|
| `feat_high` | = `inner_outs[0]` = P5 | `(B, 512, 20, 20)` |
| `feat_low` | = `reduce_outs[1]` = P4 | `(B, 512, 40, 40)` |
| `upsample_feat` = `upsample_layers[0](feat_high)` | nearest-2× upsample | `(B, 512, 40, 40)` |
| **T1** = concat along channel dim | `[upsample_feat, feat_low]` | `(B, 1024, 40, 40)` |
| **T2** = `top_down_layers[0](T1, txt_feats)` | MaxSigmoid CSP block, 8 heads × 32d | `(B, 512, 40, 40)` |

After `insert(0, ...)`: `inner_outs = [T2, P5]` = `[(B, 512, 40, 40), (B, 512, 20, 20)]`

**Iteration 2 — `idx=1`** (fuse T2 with P3):

| Step | Computation | Shape |
|---|---|---|
| `feat_high` | = `inner_outs[0]` = T2 | `(B, 512, 40, 40)` |
| `feat_low` | = `reduce_outs[0]` = P3 | `(B, 256, 80, 80)` |
| `upsample_feat` = `upsample_layers[1](feat_high)` | nearest-2× upsample | `(B, 512, 80, 80)` |
| **T3** = concat | `[upsample_feat, feat_low]` | `(B, 768, 80, 80)` |
| **T4** = `top_down_layers[1](T3, txt_feats)` | MaxSigmoid CSP block, 4 heads × 32d | `(B, 256, 80, 80)` |

After insert: `inner_outs = [T4, T2, P5]`.

**End of top-down. State:**

```
inner_outs[0] = T4  (B, 256, 80, 80)   P3 level, text-refined
inner_outs[1] = T2  (B, 512, 40, 40)   P4 level, text-refined
inner_outs[2] = P5  (B, 512, 20, 20)   P5 level, unchanged so far (it was the seed)
```

### Stage 2 — Bottom-up path

Info flows **fine → coarse** (P3 up to P5). The loop runs for `idx = 0, 1`.

```python
outs = [inner_outs[0]]   # start with P3 level
for idx in range(len(self.in_channels) - 1):   # idx: 0, 1
    feat_low = outs[-1]
    feat_high = inner_outs[idx + 1]
    downsample_feat = self.downsample_layers[idx](feat_low)
    out = self.bottom_up_layers[idx](
        torch.cat([downsample_feat, feat_high], 1), txt_feats)
    outs.append(out)
```

**Iteration 1 — `idx=0`** (fuse T4 with T2):

| Step | Computation | Shape |
|---|---|---|
| `feat_low` | = `outs[-1]` = T4 | `(B, 256, 80, 80)` |
| `feat_high` | = `inner_outs[1]` = T2 | `(B, 512, 40, 40)` |
| `downsample_feat` = `downsample_layers[0](feat_low)` | 3×3 stride-2 conv, keeps 256ch | `(B, 256, 40, 40)` |
| **T5** = concat | `[downsample_feat, feat_high]` | `(B, 768, 40, 40)` |
| **T6** = `bottom_up_layers[0](T5, txt_feats)` | MaxSigmoid CSP block, 8 heads × 32d | `(B, 512, 40, 40)` |

Appended: `outs = [T4, T6]`.

**Iteration 2 — `idx=1`** (fuse T6 with P5):

| Step | Computation | Shape |
|---|---|---|
| `feat_low` | = `outs[-1]` = T6 | `(B, 512, 40, 40)` |
| `feat_high` | = `inner_outs[2]` = P5 | `(B, 512, 20, 20)` |
| `downsample_feat` = `downsample_layers[1](feat_low)` | 3×3 stride-2 conv, keeps 512ch | `(B, 512, 20, 20)` |
| **T7** = concat | `[downsample_feat, feat_high]` | `(B, 1024, 20, 20)` |
| **T8** = `bottom_up_layers[1](T7, txt_feats)` | MaxSigmoid CSP block, 8 heads × 32d | `(B, 512, 20, 20)` |

Appended: `outs = [T4, T6, T8]`.

### Stage 3 — Out layers (no-op)

All `Identity`, so `results = outs`.

**Final outputs handed to the head:**

```
results[0] = T4:  (B, 256, 80, 80)     stride 8  — for small objects
results[1] = T6:  (B, 512, 40, 40)     stride 16 — for medium objects
results[2] = T8:  (B, 512, 20, 20)     stride 32 — for large objects
```

Same channel counts and spatial dimensions as the input P3 / P4 / P5 — the
PAFPN is shape-preserving end-to-end. What changed is *what* those channels
encode: now they've been refined twice (once top-down, once bottom-up) and
gated by text at 4 fusion points.

## Visual summary (ASCII with shapes)

```
Input tensors:
   P3 (B, 256, 80, 80)  stride 8    P4 (B, 512, 40, 40)  stride 16    P5 (B, 512, 20, 20)  stride 32
      │                                 │                                 │
      │                                 │                                 ▼
      │                                 │                                 ├── seed ──┐
      │                                 │                                 │          │
      │                                 │                         ┌───────┘          │
      │                                 │                         ▼                  │
      │                                 │                    upsample×2              │
      │                                 │                    (B, 512, 40, 40)        │
      │                                 ▼                         │                  │
      │                            ──┬──────── concat chan ───────┘                  │
      │                              │ (B, 1024, 40, 40)                             │
      │                              ▼                                                │
      │                 TD0: MaxSigmoidCSPLayerWithTwoConv  ←── txt_feats (B, N, 512)│
      │                 (attn: embed=256, 8 heads × 32d)                              │
      │                              │ (B, 512, 40, 40) = T2                          │
      │              ┌───────────────┤                                                │
      │              │               │                                                │
      │              ▼               │                                                │
      │         upsample×2           │                                                │
      │         (B, 512, 80, 80)     │                                                │
      │              │               │                                                │
      ▼              │               │                                                │
   ──┬─── concat ────┘               │                                                │
     │ (B, 768, 80, 80)              │                                                │
     ▼                               │                                                │
   TD1: MaxSigmoidCSPLayerWithTwoConv│                                                │
   (attn: embed=128, 4 heads × 32d)  │                                                │
     │ (B, 256, 80, 80) = T4         │                                                │
     ├────────────────► output P3'   │ top-down done. Now go back up.                 │
     │                               │                                                │
     ▼                               │                                                │
   downsample 3x3 s=2                │                                                │
   (B, 256, 40, 40)                  │                                                │
     │                               │                                                │
   ──┴── concat ────────────────────►┴──                                              │
         (B, 768, 40, 40)                                                             │
         ▼                                                                            │
   BU0: MaxSigmoidCSPLayerWithTwoConv ←── txt_feats                                   │
   (attn: embed=256, 8 heads × 32d)                                                   │
         │ (B, 512, 40, 40) = T6                                                      │
         ├────────────────────────────► output P4'                                    │
         │                                                                            │
         ▼                                                                            │
   downsample 3x3 s=2                                                                 │
   (B, 512, 20, 20)                                                                   │
         │                                                                            │
       ──┴── concat ─────────────────────────────────────────────────────────────────┘
             (B, 1024, 20, 20)
             ▼
   BU1: MaxSigmoidCSPLayerWithTwoConv ←── txt_feats
   (attn: embed=256, 8 heads × 32d)
             │ (B, 512, 20, 20) = T8
             └────► output P5'
```

## Inside one CSP block (TD0 as a worked example)

For **TD0** (`in_channels=1024, out_channels=512, num_blocks=3,
expand_ratio=0.5`):

```
mid_channels = out_channels * expand_ratio = 512 * 0.5 = 256

main_conv (1×1 stride=1):   1024 ch  →  2 × 256  =  512 ch     (shape-preserving)
split on channel:           512 ch  →  [half_A: 256ch, half_B: 256ch]
3 bottlenecks on half_B, each preserves 256 ch:
                            out_1: 256,  out_2: 256,  out_3: 256
attn_block (MaxSigmoidAttnBlock) on out_3:
                            256 → 256 ch      (takes txt_feats as guide)

x_main list = [half_A (256), half_B (256), out_1 (256), out_2 (256), out_3 (256), attn_out (256)]
            = 6 tensors × 256 channels each
concat:       6 × 256 = 1536 channels

final_conv (1×1 stride=1):  1536 ch  →  512 ch
```

The formula `(3 + num_blocks) * mid_channels = 6 * 256 = 1536` ✓ matches
`final_conv`'s input size. (Note: `3 + num_blocks` in
`MaxSigmoidCSPLayerWithTwoConv`, vs `2 + num_blocks` in the plain
`CSPLayerWithTwoConv` — the extra `+1` is the attention branch.)

## All four attention blocks, tabulated

For the L-640 config:

| Block | in_ch → out_ch | mid_ch | embed_ch | heads | head_ch | Spatial |
|---|---|---|---|---|---|---|
| TD0 (P5→P4) | 1024 → 512 | 256 | 256 | 8 | 32 | 40×40 |
| TD1 (P4→P3) | 768 → 256 | 128 | 128 | 4 | 32 | 80×80 |
| BU0 (P3→P4) | 768 → 512 | 256 | 256 | 8 | 32 | 40×40 |
| BU1 (P4→P5) | 1024 → 512 | 256 | 256 | 8 | 32 | 20×20 |

`head_channels = 32` for every block (that's the per-head subspace the
attention dot-products happen in). `mid_channels` and `embed_channels` are
independent hyperparameters: `mid_channels` ties to `out_channels × 0.5`,
while `embed_channels` comes from the `neck_embed_channels` config list.
For the L model they happen to be equal at each level (both 128 at P3-level,
both 256 at P4/P5-level), but that's a config choice, not a structural
requirement.

## Clarifying: `CSPLayerWithTwoConv` does NOT downsample

A common confusion: *"doesn't `CSPLayerWithTwoConv` have a 3×3 stride-2
conv at the beginning to downsample by 2?"*

**No.** The `main_conv` is a **1×1 stride-1** channel projection:

```python
# mmyolo/models/layers/yolo_bricks.py:1476-1490
self.mid_channels = int(out_channels * expand_ratio)
self.main_conv = ConvModule(
    in_channels,
    2 * self.mid_channels,
    1,                          # ← kernel_size = 1
    conv_cfg=conv_cfg,
    norm_cfg=norm_cfg,
    act_cfg=act_cfg)
self.final_conv = ConvModule(
    (2 + num_blocks) * self.mid_channels,
    out_channels,
    1,                          # ← kernel_size = 1
    ...)
```

`ConvModule` defaults `stride=1`. So both `main_conv` and `final_conv` are
1×1 channel projections. The CSP block is **completely shape-preserving
spatially**.

The two places a 3×3 stride-2 conv actually exists in the YOLO-World
architecture are both *outside* any CSP block:

1. **In the backbone** — each stage has a leading 3×3 stride-2 conv *before*
   its `CSPLayerWithTwoConv`. Built in
   `csp_darknet.py:build_stage_layer` (approximately).
2. **In the neck's bottom-up path** — the explicit `downsample_layers[idx]`
   built by `YOLOv5PAFPN.build_downsample_layer`
   (`yolov5_pafpn.py:134-150`). These run between levels, before the
   `bottom_up_layers` CSP block.

In both cases the stride-2 downsampling is an **explicit sibling** of the
CSP block, not a component of it. This is a confusing convention because
YOLOv5/v8 diagrams often draw "a stage" as a single box, but really each
stage is `Conv3x3s2 + CSPLayerWithTwoConv`.

## Observations worth keeping

- **Spatial dimensions change only in three places** in the PAFPN forward:
  the `upsample_layers` double H/W, the `downsample_layers` halve H/W, and
  the backbone is the source of the original strides. The CSP blocks
  themselves *never* change spatial dimensions.
- **Each of the 4 attention blocks sees the same `txt_feats`** `(B, N, 512)`.
  The 8 (or 4) heads give each attention block its own multi-subspace view,
  but every block starts from the same CLIP-normalized text embeddings.
  No per-level text refinement in `YOLOWorldPAFPN` (the `DualPAFPN`
  variant does refine text between top-down and bottom-up, via the pooling
  attention module).
- **The concatenation ordering differs between top-down and bottom-up**
  because of `upsample_feats_cat_first=True` (set in the `BaseYOLONeck`
  constructor). For top-down: `torch.cat([upsample_feat, feat_low], 1)`.
  For bottom-up: `torch.cat([downsample_feat, feat_high], 1)`. In both
  cases, the "new" feature (upsampled or downsampled) comes first on the
  channel dim, then the native-resolution one.
- **Total compute cost of the 4 attention blocks is dominated by the
  largest spatial map.** TD1 operates on 80×80 = 6400 pixels. Even with a
  narrower embed dim (128), that's where most of the attention-path FLOPs
  go. TD0 and BU0 operate on 40×40 = 1600 pixels. BU1 on 20×20 = 400
  pixels. So the "bottom" of the pyramid (TD1, closest to P3) is the most
  expensive one, which is also the one gating detections of *small*
  objects.

---
