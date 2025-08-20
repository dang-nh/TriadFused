# TriadFuse-Doc: Implementation Guide for a Document-Focused Adversarial Attack on VLMs

> Audience: a Cursor agent (Claude-sonnet-4.0) implementing the research prototype described in earlier discussions.
> Goal: build a **practical, reproducible** attack framework that targets **document understanding** in modern VLMs (post-2024) with **white-box → gray-box → black-box** capability, and a rigorous evaluation harness.

---

## 0) TL;DR checklist (agent)

* [ ] Create the repo scaffold (structure below).
* [ ] Set up the environment (Python 3.10+, CUDA, PyTorch 2.2+).
* [ ] Implement **surrogate adapters** for Donut, LayoutLMv3, BLIP-2, OpenCLIP (and optional Qwen2.5-VL). ([Hugging Face][1], [PyPI][2])
* [ ] Implement **EOT** (Expectation-over-Transformations) differentiable transform sampler. ([arXiv][3])
* [ ] Implement **TriadFuse** attack heads:

  * **Texture head** (low-freq DCT or multi-scale blur-robust)
  * **Glyph head** (Unicode-confusable, Gumbel-Softmax text edits) ([Unicode.org][4], [arXiv][5])
  * **Layout head** (TPS/affine micro-warps + printable “micro-stamps”) ([kornia.readthedocs.io][6])
* [ ] Implement **losses** (task-aware + perceptual LPIPS + regularizers). ([GitHub][7])
* [ ] Add **black-box query refiners** (NES / bandit style). ([Guillaume Jaume][8])
* [ ] Add **constraints** (L∞/L2, LPIPS≤τ, SSIM≥σ). ([Lightning AI][9])
* [ ] Build **dataset loaders** (FUNSD, CORD, SROIE, DocVQA-Task3) & PDF raster I/O. ([kornia.readthedocs.io][10], [GitHub][11], [rrc.cvc.uab.es][12])
* [ ] Implement **evaluation** (attack success, query count, CER/WER deltas, answer flips, transfer). ([PyPI][13])
* [ ] Provide **CLI scripts** to run attacks and export reports.

---

## 1) Repository structure

```
triadfuse/
├─ README.md
├─ pyproject.toml            # or requirements.txt
├─ triadfuse/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ eot.py                 # EOT sampler + Albumentations/Kornia wrappers
│  ├─ constraints.py         # norm, SSIM/LPIPS guards
│  ├─ losses.py              # composite objective
│  ├─ utils/
│  │  ├─ pdfio.py            # PyMuPDF raster, DPI control
│  │  ├─ vis.py              # debug renders, grids
│  │  ├─ textops.py          # homoglyph tables, tokenizer helpers
│  ├─ heads/
│  │  ├─ texture.py          # low-freq DCT param / multi-scale blur-robust
│  │  ├─ glyph.py            # Unicode-confusable + Gumbel-Softmax edits
│  │  ├─ layout.py           # TPS/affine micro-stamps overlay (Kornia)
│  ├─ surrogate/
│  │  ├─ base.py
│  │  ├─ donut.py            # Hugging Face Donut
│  │  ├─ layoutlmv3.py       # Hugging Face LayoutLMv3
│  │  ├─ blip2.py            # Hugging Face BLIP-2
│  │  ├─ openclip.py         # OpenCLIP for retrieval/semantic similarity
│  │  ├─ qwen25vl.py         # (optional) Qwen2.5-VL inference adapter
│  ├─ refine/
│  │  ├─ nes.py              # NES/Bandit gradient-free refinement
│  ├─ data/
│  │  ├─ funsd.py
│  │  ├─ cord.py
│  │  ├─ sroie.py
│  │  ├─ docvqa_t3.py
├─ scripts/
│  ├─ install.sh
│  ├─ attack_train.py        # white/gray-box optimization
│  ├─ attack_refine.py       # black-box query refiner
│  ├─ eval_attack.py         # batch evaluation, metrics
│  ├─ demo_notebook.ipynb
└─ tests/
   ├─ test_eot.py
   ├─ test_heads.py
   ├─ test_losses.py
   ├─ test_eval.py
```

---

## 2) Environment & dependencies

**Base**: Python 3.10+, CUDA-enabled PyTorch 2.2+, TorchVision.

Install (Ubuntu-like):

```bash
# Core
pip install "torch>=2.2" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Vision & transforms
pip install kornia albumentations opencv-python-headless

# Models
pip install transformers accelerate bitsandbytes open_clip_torch sentence-transformers

# OCR & doc I/O
pip install paddleocr easyocr pymupdf

# Metrics
pip install lpips pytorch-msssim jiwer fastwer

# Utilities
pip install rich tyro einops
```

* Use **bitsandbytes** 8-bit/4-bit when loading large VLMs; `BitsAndBytesConfig(load_in_8bit=True)` works with `transformers` and `accelerate` `device_map="auto"` to fit on commodity GPUs. ([Hugging Face][14])
* For PDF rasterization, use **PyMuPDF (fitz)**: page → pixmap(DPI) → numpy. ([PyMuPDF][15])

---

## 3) Surrogate model adapters (white/gray-box)

Implement `triadfuse/surrogate/base.py` with a common interface:

```python
class SurrogateModel(nn.Module):
    def forward_task_loss(self, image, prompt, target):
        """Return scalar task loss and optional dict of aux features."""
    def tokenize(self, prompt): ...
    @torch.no_grad()
    def predict(self, image, prompt): ...
```

Adapters to implement:

* **Donut** (OCR-free doc VQA/IE): Hugging Face version has processors & generation APIs. Use for text-Only outputs (JSON strings). ([Hugging Face][1], [GitHub][16])
* **LayoutLMv3** (Document AI backbone with unified text+image masking): use token classification or VQA heads for FUNSD/CORD. ([Hugging Face][17])
* **BLIP-2** (frozen encoders + Q-Former): use image-to-text VQA loss. ([Hugging Face][18])
* **OpenCLIP** (image/text encoders): use cosine similarity to guide semantic drift (answer gating / retrieval). ([PyPI][2])
* **(Optional) Qwen2.5-VL** for modern VLM behavior; load via `transformers` model docs. ([Hugging Face][19])

> Tip: when VRAM is tight, quantize LLM components with bitsandbytes and rely on `accelerate` automatic device mapping. ([Hugging Face][20])

---

## 4) Expectation-over-Transformations (EOT) module

Create `triadfuse/eot.py` with a sampler that applies **differentiable** transforms drawn from a distribution that mimics printing/scanning and UI rendering:

* Geom: random **affine**, **perspective**, small **TPS** warps (Kornia). ([kornia.readthedocs.io][6])
* Photometric: **JPEG/WebP compression**, gamma/contrast, Gaussian/motion blur (Albumentations + Kornia filters). ([Albumentations][21], [vfdev-5-albumentations.readthedocs.io][22], [kornia.readthedocs.io][23])
* Resolution: random **down/up-sample**, DPI jitter (PDF raster). ([PyMuPDF][15])

Why: EOT optimizes expectation of the attack objective over realistic transforms → **physical/world robustness** and **defense randomization resistance**. ([arXiv][3])

**Skeleton:**

```python
class EOT(nn.Module):
    def __init__(self, tps=True, jpeg=True, blur=True, n_samples=8, seed=None): ...
    def sample(self, x):
        # return list[transformed_x], and a callable to backprop through
    def __call__(self, x, fn_loss):
        losses = []
        for _ in range(self.n):
            xt = self._transform_once(x)
            losses.append(fn_loss(xt))
        return torch.stack(losses).mean()
```

---

## 5) TriadFuse attack heads

### 5.1 Texture head (low-frequency, blur-robust)

**Idea:** Optimize a compact set of **low-frequency** coefficients (global or tiled) to keep perturbations minimally visible but robust under downsampling/compression.

* Parameterize either:

  1. **DCT-low-band** coefficients (use `torch_dct` or implement via FFT basis) and inverse-transform to RGB; or
  2. **Multi-scale smooth fields** optimized pre-blurred with Gaussian/motion blur to survive printers/scanners. ([kornia.org][24])

**Why:** JPEG, resampling, and small blur kill high-freq noise; low-freq survives and transfers better across VLM visual encoders.

### 5.2 Glyph head (Unicode-confusable & typographic micro-edits)

**Idea:** For **PDF/HTML/native text**, produce adversarial **text** changes that visually preserve meaning to humans (homoglyphs, small kerning/spacing) but perturb OCR/LLM tokenization.

* Use **Unicode confusables** tables to build substitution candidates (Latin↔Greek/Cyrillic look-alikes). Implement a **Gumbel-Softmax** selection layer to make the discrete choice **differentiable** during white-box optimization. ([Unicode.org][4], [PyTorch Docs][25])
* Optionally add zero-width joins or punctuation variants conservatively (respect rendering constraints).

**Why:** Document VLMs often route through OCR or text tokens internally; tiny text-level changes can flip extraction/answers while appearing the same to humans. Standards acknowledge confusable risks (UTS#39). ([Unicode.org][4])

### 5.3 Layout head (TPS micro-warps & micro-stamps)

**Idea:** Add tiny, printable **micro-stamps** (e.g., faint dots/diacritics) and **local TPS warps** along table lines/field boundaries to desynchronize layout cues (reading order, bbox alignments) while minimally affecting legibility.

* Implement **TPS warp** (Kornia) with control points restricted to grid intersections / whitespace. ([kornia.readthedocs.io][6])
* Stamps: alpha-blend small patterns placed via differentiable coordinates; constrain area budget (<0.2% pixels).

**Why:** Layout-centric models (LayoutLMv3) and OCR detectors rely on geometry; sub-pixel warps can disturb bounding boxes/ordering without visible artifacts.

---

## 6) Composite objective (white/gray-box)

Given clean input **x**, prompt **p**, target (label/answer) **y**, perturbation parameters **θ = {θ\_tex, θ\_glyph, θ\_layout}**, and EOT distribution **T**:

$$
\min_{\theta}\; \mathbb{E}_{t \sim T}\; 
\Big[ \underbrace{\mathcal{L}_{task}\big(f(t(\hat{x}_\theta), p), y\big)}_{\text{VQA/IE or retrieval loss}}
+ \lambda_{feat}\,\underbrace{D\big(\phi_k(t(\hat{x}_\theta)), \phi_k(t(x))\big)}_{\text{feature-targeted (Q-Former / CLIP layer)}} \Big] 
+ \lambda_{per}\,\underbrace{\text{LPIPS}(t(\hat{x}_\theta), t(x))}_{\text{perceptual budget}} 
+ \lambda_{tv}\,TV(\delta) 
$$

subject to $\|\delta\|_\infty \le \epsilon$, SSIM$(\hat{x}_\theta, x)\ge \sigma$, area$_{\text{stamps}}\le \alpha$.

* **L\_task**:

  * Donut/LayoutLMv3: cross-entropy on tokens/answers.
  * BLIP-2: negative log-likelihood of target string or contrastive with OpenCLIP if supervision is weak. ([Hugging Face][18], [PyPI][2])
* **D(·)**: layer-wise feature drift to steer toward failure modes, e.g., **Q-Former** activations in BLIP-2 or **CLIP** penultimate embeddings. ([Hugging Face][18], [PyPI][2])
* **LPIPS** perceptual constraint and **SSIM** guard. ([GitHub][7], [Lightning AI][9])

---

## 7) Constraints & safety rails

* **L∞/L2 norm clamps**, **LPIPS ≤ τ**, **SSIM ≥ σ** (tunable). ([Lightning AI][9])
* **Text safety**: enforce replacement only from confusable tables; avoid functional meaning changes. ([Unicode.org][4])
* **Printability**: micro-stamps with min alpha and dwell time; preview through **PDF→raster** at \[150–300] DPI EOT. ([PyMuPDF][15])

---

## 8) Black-box query refiner (NES/Bandit)

When gradients are unavailable (closed VLM APIs), run a few **white/gray-box** steps on surrogates, then **refine** with **NES**:

* Sample δ directions $u_i \sim \mathcal{N}(0,I)$, estimate ∇ via symmetric queries:
  $\hat{g} \propto \frac{1}{m} \sum_i \big( L(x+\sigma u_i) - L(x-\sigma u_i)\big) u_i$.
* Update θ (or directly pixels if allowed) with momentum & projection onto budgets.
* Early-stop on success; track **query count**. ([Guillaume Jaume][8])

---

## 9) Datasets & I/O

Provide lightweight loaders:

* **FUNSD** (forms understanding): forms, entities, relations. ([kornia.readthedocs.io][10])
* **CORD** (receipts, Korean/Indonesian, key-information extraction). ([GitHub][11])
* **SROIE** (ICDAR’19 scanned receipts; OCR/IE tasks). ([rrc.cvc.uab.es][12])
* **DocVQA-Task 3 (Infographic/InfoVQA)** for chart/infographic Q\&A. ([Albumentations][26])

Support **PDFs** via PyMuPDF for rasterization at multiple DPIs (EOT). ([PyMuPDF][15])

---

## 10) Evaluation suite

Metrics to implement:

* **Attack Success Rate (ASR)**: fraction of items where answer/IE fields flip to target or away from ground truth.
* **Query count** (black-box).
* **Transfer**: success when crafted on surrogate A and tested on B (e.g., Donut→Qwen2.5-VL). ([Hugging Face][19])
* **OCR sensitivity**: compute **CER/WER** between clean vs. attacked OCR text with `jiwer` / `fastwer`. ([PyPI][13])
* **Perceptual budget**: LPIPS, SSIM. ([GitHub][7], [Lightning AI][9])

Export a JSON/CSV report with per-item and aggregate stats.

---

## 11) High-level implementation outline (key modules)

### 11.1 EOT transforms (simplified)

```python
# triadfuse/eot.py
import kornia as K
import albumentations as A
import torch, random

class EOT:
    def __init__(self, n=8, jpeg=True, tps=True, blur=True, seed=0):
        self.n = n; random.seed(seed)
        self.jpeg = A.ImageCompression(quality_lower=40, quality_upper=80, p=0.9)  # JPEG/WebP in Albumentations
        self.photometric = A.Compose([
            A.RandomBrightnessContrast(0.1,0.1,p=0.7),
            A.GaussianBlur(blur_limit=(3,5), p=0.5),
        ])
        self.tps_grid = K.geometry.transform.ThinPlateSpline # use Kornia TPS API

    def _apply_tps(self, x):
        # construct mild control points, warp with Kornia TPS
        # returns x_warped
        ...

    def _alb(self, x):
        # x: BCHW float in [0,1]; albumentations expects HWC uint8/float32
        ...

    def sample_once(self, x):
        xt = x
        if random.random() < 0.7: xt = self._apply_tps(xt)
        xt = self._alb(xt)
        return xt

    def expectation(self, x, loss_fn):
        losses = []
        for _ in range(self.n):
            losses.append(loss_fn(self.sample_once(x)))
        return torch.stack(losses).mean()
```

(Albumentations JPEG/ImageCompression and Kornia TPS/filters are documented; use the referenced APIs.) ([Albumentations][21], [kornia.readthedocs.io][6], [explore.albumentations.ai][27])

### 11.2 Texture head (low-freq parameterization)

```python
# triadfuse/heads/texture.py
class TextureHead(nn.Module):
    def __init__(self, shape, lf_ratio=0.25):
        super().__init__()
        H, W = shape[-2:]
        self.Hl, self.Wl = int(H*lf_ratio), int(W*lf_ratio)
        self.coeff = nn.Parameter(torch.zeros(1,3,self.Hl,self.Wl))

    def forward(self, base):
        # upsample low-freq coeffs smoothly to image size
        pert = F.interpolate(self.coeff, size=base.shape[-2:], mode='bicubic', align_corners=False)
        return torch.clamp(base + pert.tanh()*0.01, 0, 1)
```

(Optionally replace upsample with IDCT/torch\_dct if available.)

### 11.3 Glyph head (Unicode-confusable with Gumbel-Softmax)

```python
# triadfuse/heads/glyph.py
import torch.nn.functional as F

class GlyphHead:
    def __init__(self, confusable_map):
        self.table = confusable_map   # dict: ascii_char -> [candidates]
        # prebuild logits per position; size: [L, K_i] variable -> pad to Kmax

    def sample_edit(self, logits_row, tau=0.5, hard=True):
        sel = F.gumbel_softmax(logits_row, tau=tau, hard=hard)  # differentiable choice
        return sel

    def render_text(self, text, logits):
        # build edited string using selected confusables while keeping appearance
        ...
```

(Confusables: per UTS#39; differentiable selection via PyTorch `gumbel_softmax`.) ([Unicode.org][4], [PyTorch Docs][25])

### 11.4 Layout head (TPS micro-stamps)

```python
# triadfuse/heads/layout.py
class MicroStampHead(nn.Module):
    def __init__(self, max_area=0.002):
        super().__init__()
        self.xy_alpha = nn.Parameter(torch.rand(16,3))  # [N, (x,y,alpha)]
        self.patch = nn.Parameter(torch.zeros(1,1,7,7)) # tiny stamp kernel

    def forward(self, x):
        B,C,H,W = x.shape
        out = x
        for (cx,cy,a) in self.xy_alpha.sigmoid():
            px, py = int(cx*W), int(cy*H)
            out[:,:,py-3:py+4, px-3:px+4] = \
                out[:,:,py-3:py+4, px-3:px+4]*(1-a) + self.patch.tanh()*a
        return torch.clamp(out,0,1)
```

Add optional Kornia **TPS** warp around detected table lines.

### 11.5 Composite loss & loop

```python
# triadfuse/attack_train.py (core loop)
for step in range(max_steps):
    def loss_once(x_adv):
        L_task, feats = surrogate.forward_task_loss(x_adv, prompt, target)
        L_feat = feature_divergence(feats, feats_clean)
        L_per  = lpips(x_adv, x_clean)
        return L_task + lam_f*L_feat + lam_p*L_per

    L = eot.expectation(x_adv, loss_once)
    opt.zero_grad(); L.backward(); opt.step()
    project_constraints(x_adv, x_clean)  # norms, SSIM guard
```

### 11.6 NES refiner (black-box)

```python
# triadfuse/refine/nes.py
def nes_step(x, loss_fn, sigma=0.5/255, m=40, step=1.0/255):
    u = torch.randn(m, *x.shape).to(x.device)
    fpos = batch_eval(loss_fn, x + sigma*u)
    fneg = batch_eval(loss_fn, x - sigma*u)
    g = ((fpos - fneg).view(m,1,1,1,1) * u).mean(0) / (2*sigma)
    x_new = x + step * g.sign()
    return x_new.clamp(0,1)
```

(NES for query-efficient gradient estimation.) ([Guillaume Jaume][8])

---

## 12) Evaluation scripts

`eval_attack.py` should:

1. Run targeted/untargeted attacks on a split.
2. Collect **answers** from each surrogate + a held-out VLM (if available).
3. Compute:

   * **ASR** (per task),
   * **queries** (black-box),
   * **LPIPS/SSIM**,
   * **OCR CER/WER** deltas using `jiwer`/`fastwer`. ([PyPI][13])
4. Save `results.json` and a Markdown summary.

---

## 13) Example CLI

```bash
# White/gray-box on Donut + LayoutLMv3 with EOT
python scripts/attack_train.py \
  --model donut --dataset funsd --prompt "extract fields: ..." \
  --eot.samples 8 --head.texture --head.glyph --head.layout \
  --lpips 0.05 --ssim 0.95 --steps 500

# Black-box refinement against Qwen2.5-VL endpoint (assume wrapper)
python scripts/attack_refine.py \
  --bb_model qwen25vl --queries 200 --sigma 0.5 --step 1.0 \
  --init from_surrogate_artifacts

# Evaluate + report
python scripts/eval_attack.py --dataset sroie --report out/report.json
```

---

## 14) Testing

* Unit tests for EOT determinism under fixed seed, constraints projection, and loss gradients.
* Golden cases for glyph substitutions on a small template PDF (rendered, OCR text compared).

---

## 15) Ethical use

This code is for **research**, **red-teaming**, and **hardening** multimodal systems in **document** workflows. Do **not** deploy against systems without authorization. Follow best practices for evaluating robustness; avoid “obscure-by-randomness” defenses. ([arXiv][28])

---

## 16) Appendices

### A) Surrogate quickstarts (snippets)

**BLIP-2** (HF Transformers):

```python
from transformers import Blip2Processor, Blip2ForConditionalGeneration
proc = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", device_map="auto")
```

(Architecture uses frozen encoders + Q-Former.) ([Hugging Face][18])

**LayoutLMv3**:

```python
from transformers import LayoutLMv3Processor, LayoutLMv3ForQuestionAnswering
proc = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
model = LayoutLMv3ForQuestionAnswering.from_pretrained("microsoft/layoutlmv3-base", device_map="auto")
```

(Pretrained for Document AI with unified text+image masking.) ([Hugging Face][17])

**Donut**:

```python
from transformers import DonutProcessor, VisionEncoderDecoderModel
proc = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base", device_map="auto")
```

(Donut is OCR-free doc VQA/IE.) ([Hugging Face][1])

**OpenCLIP**:

```python
import open_clip
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
```

(Open-source CLIP implementation.) ([PyPI][2])

**Qwen2.5-VL** (optional):

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
proc = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(..., device_map="auto")
```

(Recent multimodal VLM line with HF integration.) ([Hugging Face][19])

### B) OCR & metrics

**PaddleOCR** quickstart (Python API):

```python
from paddleocr import PaddleOCR
ocr = PaddleOCR(lang="en")
res = ocr.ocr(img_np)
```

([PaddlePaddle][29])

**CER/WER** with `jiwer`:

```python
from jiwer import cer, wer
delta_cer = cer(ref_text, adv_text) - cer(ref_text, clean_text)
```

([PyPI][13])

**LPIPS**:

```python
import lpips
loss_fn = lpips.LPIPS(net='alex').to(device)
val = loss_fn(x_adv, x_clean)
```

([GitHub][7])

### C) Key references backing design choices

* **EOT** for transform-robust attacks (also to handle randomized defenses). ([arXiv][3])
* **Kornia TPS/filters** for differentiable warps & blurs. ([kornia.readthedocs.io][6])
* **Albumentations JPEG/ImageCompression** to simulate lossy pipelines. ([vfdev-5-albumentations.readthedocs.io][22], [explore.albumentations.ai][27])
* **BLIP-2** frozen encoders + Q-Former (feature-level objectives). ([Hugging Face][18])
* **Donut** OCR-free doc modeling. ([Hugging Face][1])
* **LayoutLMv3** for layout-aware surrogates. ([Hugging Face][17])
* **OpenCLIP** for semantic drift gating. ([PyPI][2])
* **NES** black-box gradient estimation. ([Guillaume Jaume][8])
* **Unicode confusables** (UTS#39) supporting glyph edits. ([Unicode.org][4])
* **PDF rasterization** pipeline via PyMuPDF. ([PyMuPDF][15])

---

## 17) Agent plan of work (fine-grained)

1. **Bootstrap**

   * [ ] Generate repo from the structure.
   * [ ] Add `scripts/install.sh` installing pinned versions; smoke-test imports.

2. **Models**

   * [ ] Implement `surrogate/base.py`, then `donut.py`, `layoutlmv3.py`, `blip2.py`, `openclip.py`, `qwen25vl.py` (optional).
   * [ ] Each adapter: `forward_task_loss`, `predict`, `tokenize`, unit tests on toy images.

3. **EOT**

   * [ ] Implement Albumentations photometric stack + Kornia TPS & blurs.
   * [ ] Unit test: with fixed seed, mean(loss) over N samples is stable (±ε).

4. **Heads**

   * [ ] `texture.py`: low-freq param & clamp.
   * [ ] `glyph.py`: build confusable map from a shipped JSON (subset of UTS#39 tables); implement Gumbel-Softmax selection; renderer for PDF/bitmap. ([Unicode.org][4], [PyTorch Docs][25])
   * [ ] `layout.py`: micro-stamps overlay + optional TPS around detected lines.

5. **Losses & constraints**

   * [ ] Implement composite objective with tunable λ’s; integrate **LPIPS** and **SSIM** checks. ([GitHub][7], [Lightning AI][9])

6. **Training/optimization**

   * [ ] `attack_train.py` (Adam/PGD hybrid); supports targeted/untargeted.
   * [ ] Export artifacts: δ maps, edited text, placement metadata.

7. **Black-box**

   * [ ] `refine/nes.py`: implement NES loop with batching; early-stop on success.

8. **Data & I/O**

   * [ ] FUNSD/CORD/SROIE/DocVQA loaders; **pdfio.py** with PyMuPDF raster at multiple DPIs. ([kornia.readthedocs.io][10], [GitHub][11], [rrc.cvc.uab.es][12])

9. **Evaluation**

   * [ ] `eval_attack.py`: ASR, transfer matrix, LPIPS/SSIM, **CER/WER** deltas. ([PyPI][13])
   * [ ] Save Markdown + JSON with tables & examples.

10. **Docs & examples**

* [ ] Populate `README.md` with usage examples and caveats.
* [ ] Add a demo notebook attacking a handful of receipts/forms.

---

## 18) Notes & assumptions

* This guide assumes access to GPUs; for larger VLMs use 8-bit loading via **bitsandbytes** + **accelerate** auto device mapping. ([Hugging Face][14])
* For Qwen2.5-VL/InternVL/Kimi-VL targets, prioritize **black-box** refinement after surrogate crafting; adapters can still be added as capabilities permit. ([Hugging Face][19], [GitHub][30], [arXiv][31])
* Validate against multiple raster DPIs and print-scan loops to ensure robustness (EOT).