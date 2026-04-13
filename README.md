GPT-from-Scratch: Generative Transformer Implementation
A from-scratch implementation of a GPT-style Decoder-only Transformer architecture, developed for the M.Sc. Cognitive Systems program at the University of Potsdam. This project demonstrates the fundamental mechanics of Large Language Models (LLMs), including causal self-attention, layer normalization, and advanced decoding strategies.


## Project structure

```text
gpt2-from-scratch/
├── model/
│   └── model_gpt.py    # Core architecture (Multi-head Attention, Transformer Blocks)
├── data/               # ← gitignored; place Frankenstein.txt here
├── utils.py            # Tokenization (tiktoken) and decoding logic
├── train.py            # Training script with evaluation loops
├── assignment4.py      # Main entry point for inference and evaluation
├── config.json         # Model configuration parameters
├── Dockerfile          # Containerization for environment reproducibility
├── requirements.txt    # Project dependencies
└── .gitignore          # Rules to exclude weights (.pth) and .venv
```

---

## 1. Set up the environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Python ≥ 3.10 recommended.

---

## 2. Obtain and place the dataset
### Place it in the data/ directory:

data/Frankenstein.txt

---

## 3. Run the experiments
### Model Training

```bash
python train.py
```

### Text Generation (Inference)

```bash
# Default generation (using settings in config.json)
python assignment4.py
```

---

## 4. Technical Features & Deliverables

| Artifact | Description |
|:---------|:------------|
| `model/model_gpt.py` | Implementation of Causal Self-Attention and Feed-Forward blocks |
| `utils.py` | Logic for Temperature scaling, Top-k, and Nucleus (Top-p) sampling |
| `loss-plot.png` | Visualization of training and validation loss over epochs |
| `Dockerfile` | Multi-platform build instructions for GPU/CPU parity |

---

## 5. Configuration defaults

All architectural defaults live in `config.json`:

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `n_layer` | 12 | Number of Transformer blocks |
| `n_head` | 12 | Number of attention heads |
| `n_embd` | 768 | Embedding dimension size |
| `block_size` | 1024 | Maximum sequence length (context window) |
| `top_p` | 0.9 | Probability threshold for Nucleus sampling |

---

## 6. Reproducibility

- `torch.manual_seed(42)` is used throughout to ensure deterministic behavior across training runs.
- The provided `Dockerfile` allows for building a consistent environment regardless of host OS (Linux/Windows/macOS).
- `torch.backends.mps.is_available()` is checked to ensure hardware acceleration on Apple Silicon (M-series), while `torch.cuda.is_available()` is used for NVIDIA hardware.
- `torch.backends.cudnn.deterministic = True` is set when running on CUDA to ensure reproducibility of convolution/attention layers.

---

## References

- Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2).
- Advanced Natural Language Processing Course, University of Potsdam.
- OpenAI `tiktoken` documentation for BPE tokenization strategies.









