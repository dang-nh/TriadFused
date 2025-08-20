# TriadFuse

**TriadFuse** is a compound adversarial attack framework for document Vision-Language Models (VLMs). It combines multiple attack modalities (texture, glyph, layout) with Expectation-over-Transformations (EOT) to create robust adversarial perturbations for document images.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/triadfuse.git
cd triadfuse

# Install the package
pip install -e .

# Or use the install script
bash scripts/install.sh
```

### Basic Usage

Attack a document image with the default Donut model:

```bash
python scripts/attack_train.py \
    --input path/to/document.png \
    --target "9999" \
    --steps 200 \
    --output outputs/
```

This will:
1. Load your document image
2. Optimize adversarial perturbations using the texture head
3. Apply EOT transformations for robustness
4. Save the adversarial image and metrics to `outputs/`

## 📋 Features (MVP)

### Current Implementation
- ✅ **Texture Head**: Low-frequency imperceptible perturbations
- ✅ **EOT Module**: Expectation-over-Transformations for robustness
- ✅ **Donut Surrogate**: OCR-free document understanding model
- ✅ **Constraint Enforcement**: L∞, SSIM, and LPIPS constraints
- ✅ **CLI Interface**: Easy-to-use command-line tool

### Coming in v2
- 🔄 **Glyph Head**: Unicode-confusable text manipulations
- 🔄 **Layout Head**: TPS warps and micro-stamp overlays
- 🔄 **Additional Surrogates**: LayoutLMv3, BLIP-2, OpenCLIP
- 🔄 **Black-box Refinement**: NES/Bandit query-efficient attacks
- 🔄 **Full Dataset Support**: FUNSD, CORD, SROIE, DocVQA

## 🎯 Example Attack

```python
from triadfuse import EOT, TextureHead, DonutSurrogate
from triadfuse.utils.seed import set_seed
import torch
from PIL import Image
import torchvision.transforms as T

# Set random seed for reproducibility
set_seed(1337)

# Load image
img = Image.open("invoice.png").convert("RGB")
x = T.ToTensor()(img).unsqueeze(0).cuda()

# Initialize components
texture_head = TextureHead(img_hw=x.shape[-2:], scale=0.01).cuda()
eot = EOT(n=4)  # 4 transformation samples
model = DonutSurrogate(device="cuda")

# Define attack objective
prompt = "<s_docvqa><s_question> What is the total? <s_answer>"
target = "0.00"

# Optimize attack
optimizer = torch.optim.Adam(texture_head.parameters(), lr=0.05)

for step in range(100):
    def loss_fn(x_t):
        x_adv = texture_head(x_t)
        loss, _ = model.forward_task_loss(x_adv, prompt, target)
        return loss
    
    loss = eot.expectation(x, loss_fn)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Get adversarial image
x_adv = texture_head(x)
```

## 🛠️ CLI Options

### Main Attack Script

```bash
python scripts/attack_train.py --help
```

Key options:
- `--input`: Path to input document image
- `--target`: Target text to induce
- `--prompt`: Task prompt for the model
- `--steps`: Number of optimization steps (default: 200)
- `--eps`: Maximum L∞ perturbation (default: 2/255)
- `--ssim_min`: Minimum SSIM constraint (default: 0.95)
- `--eot_samples`: Number of EOT transformation samples (default: 4)

### Example Commands

**Basic document attack:**
```bash
python scripts/attack_train.py \
    --input receipt.jpg \
    --prompt "<s_docvqa><s_question> What is the date? <s_answer>" \
    --target "2099-12-31" \
    --steps 150
```

**With stronger constraints:**
```bash
python scripts/attack_train.py \
    --input form.png \
    --target "APPROVED" \
    --eps 0.005 \
    --ssim_min 0.98 \
    --lpips_weight 0.1
```

**Fast testing (fewer steps and EOT samples):**
```bash
python scripts/attack_train.py \
    --input test.png \
    --target "test" \
    --steps 50 \
    --eot_samples 2 \
    --max_img_size 512
```

## 📊 Output Structure

After running an attack, you'll find:

```
outputs/
├── adversarial.png   # Adversarial image
├── clean.png        # Original image
├── difference.png   # Visualization of perturbation
└── results.json     # Metrics and predictions
```

The `results.json` contains:
- Attack success status
- Clean and adversarial predictions
- Perceptual metrics (SSIM, LPIPS)
- Norm metrics (L∞, L2)
- Constraint satisfaction

## 🧪 Testing

Run the test suite:

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/ -v

# Run specific test
pytest tests/test_eot.py -v
```

## 📚 Architecture

```
TriadFuse
├── EOT Module          # Expectation-over-Transformations
│   ├── JPEG compression
│   ├── Random resize
│   ├── Gaussian blur
│   └── Gamma correction
│
├── Attack Heads
│   ├── Texture Head    # Low-frequency perturbations ✅
│   ├── Glyph Head     # Text manipulations (TODO)
│   └── Layout Head    # Geometric distortions (TODO)
│
├── Surrogate Models
│   ├── Donut          # OCR-free document VQA ✅
│   ├── LayoutLMv3     # Layout-aware model (TODO)
│   └── BLIP-2         # Vision-language model (TODO)
│
└── Constraints
    ├── L∞/L2 norms
    ├── SSIM similarity
    └── LPIPS perceptual
```

## 🔧 Development

### Project Structure

```
triadfuse/
├── triadfuse/          # Main package
│   ├── eot.py         # EOT implementation
│   ├── heads/         # Attack heads
│   ├── surrogate/     # Model adapters
│   ├── constraints.py # Constraint functions
│   └── losses.py      # Loss functions
├── scripts/           # CLI scripts
├── tests/            # Test suite
└── docs/             # Documentation
```

### Adding a New Surrogate Model

1. Create a new file in `triadfuse/surrogate/`
2. Inherit from `SurrogateModel` base class
3. Implement required methods:
   - `forward_task_loss()`
   - `predict()`

### Adding a New Attack Head

1. Create a new file in `triadfuse/heads/`
2. Inherit from `nn.Module`
3. Implement the `forward()` method
4. Add constraint projection if needed

## 📖 Citation

If you use TriadFuse in your research, please cite:

```bibtex
@software{triadfuse2024,
  title = {TriadFuse: Compound Adversarial Attack Framework for Document VLMs},
  author = {TriadFuse Team},
  year = {2024},
  url = {https://github.com/yourusername/triadfuse}
}
```

## ⚠️ Ethical Considerations

This framework is intended for:
- **Research** on VLM robustness
- **Red-teaming** document AI systems
- **Improving** model defenses

**Do not** use this tool to attack systems without authorization.

## 📝 License

MIT License - see LICENSE file for details.

## 🚧 Roadmap

### v1.0 (Current MVP)
- [x] Texture head implementation
- [x] EOT robustness module
- [x] Donut surrogate adapter
- [x] Basic CLI interface

### v2.0 (Planned)
- [ ] Glyph head with Unicode confusables
- [ ] Layout head with TPS warps
- [ ] Multiple surrogate models
- [ ] Black-box query refinement
- [ ] Full dataset support
- [ ] Advanced evaluation metrics

### v3.0 (Future)
- [ ] Multi-modal attack coordination
- [ ] Adaptive attack strategies
- [ ] Defense evaluation toolkit
- [ ] Web interface

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📬 Contact

For questions or collaboration:
- Open an issue on GitHub
- Email: contact@triadfuse.org

---

**Note**: This is an MVP implementation focusing on the texture head attack with EOT robustness. The glyph and layout heads are stubbed for future development.