import os                                      # filesystem helpers
import urllib.request                          # simple HTTP download
from collections import OrderedDict            # for state_dict remapping
from model.model_gpt import GPTModel           # GPT architecture
import tiktoken                                # GPT-2 tokenizer
import torch                                   # tensor/runtime utilities
from train import generate_and_print_sample    # helper to sample text


def _remap_causal_mask_keys(state_dict):
    """Remap 'att.causal_mask' <-> 'att.mask' to load both local and pretrained checkpoints."""
    new_state = OrderedDict()
    for k, v in state_dict.items():
        if "att.causal_mask" in k:
            new_state[k.replace("att.causal_mask", "att.mask")] = v
        else:
            new_state[k] = v
    return new_state


def load_pretrained_gpt(start_context):
     # Define GPT-2 small model configuration parameters
    gpt_config = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": True,
    }

    # Initialize model and tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")  # same tokenizer as training
    model = GPTModel(gpt_config)   

    # Choose device: prefer MPS (Apple GPU) if available, else CPU
    if torch.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    file_name = "gpt2-small-124M.pth"                                  # checkpoint name
    url = f"https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/{file_name}"
    local_path = os.path.join(".", file_name)                          # cache path

    # Download if missing
    if not os.path.exists(local_path):
        print(f"[PRETRAINED] Downloading pretrained GPT-2 small model from {url}...")
        urllib.request.urlretrieve(url, local_path)
        print(f"[PRETRAINED] Downloaded and saved to {local_path}")
    else:
        print(f"[PRETRAINED] Using cached weights at {local_path}")

   

                                   # rebuild model skeleton

    # Load pretrained weights into model
    state = torch.load(local_path, map_location=device)            # load checkpoint to device
    state = _remap_causal_mask_keys(state)                         # align buffer naming
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Generate sample text from the pretrained model
    print("[PRETRAINED] Generating sample text...")
    generate_and_print_sample(model, tokenizer, device, start_context)