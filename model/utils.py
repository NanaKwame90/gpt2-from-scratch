import tiktoken                                     # GPT-2 tokenizer
import torch                                        # tensor operations
from torch.utils.data import Dataset, DataLoader    # dataset/dataloader wrappers

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize entire text corpus into GPT-2 IDs
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Sliding window to create input/target pairs shifted by one token
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]   # returns paired sequences


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")                    # init tokenizer
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)   # build dataset
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader

def generate_text_simple(model, idx, max_new_tokens, context_size):
    # Greedy decoding: always choose argmax token
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]                     # keep only supported context
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]                             # logits for next position
        idx_next = torch.argmax(logits, dim=-1, keepdim=True) # greedy selection
        idx = torch.cat((idx, idx_next), dim=1)               # append choice
    return idx


def decode_1(model, idx, max_new_tokens, context_size, k=50):
    """
    Top-k sampling (k=20 default). Inspired by Raschka "LLMs from Scratch" ch. 5
    and PyTorch multinomial docs for sampling from categorical distributions.

    Args:
        model: A trained language model.
        idx: Initial token indices.
        max_new_tokens: Number of new tokens to generate.
        context_size: Sliding window context size.

    Returns:
        Generated token indices including the new tokens.
    """
    model.eval()

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        k_eff = min(k, logits.shape[-1])
        top_logits, top_idx = torch.topk(logits, k_eff, dim=-1)

        probs = torch.softmax(top_logits, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1)
        idx_next = top_idx.gather(-1, sampled)

        idx = torch.cat((idx, idx_next), dim=1)

    model.train()
    return idx


def decode_2(model, idx, max_new_tokens, context_size, p=0.9, temperature=0.9):
    """Nucleus (top-p) sampling with temperature. Inspired by Holtzman et al., 2019, "The Curious Case of Neural Text Degeneration", https://arxiv.org/abs/1904.09751
    """
    model.eval()

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)

        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)

        cutoff = cumulative > p
        cutoff[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(cutoff, 0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

        sampled = torch.multinomial(sorted_probs, num_samples=1)
        idx_next = sorted_idx.gather(-1, sampled)

        idx = torch.cat((idx, idx_next), dim=1)

    model.train()
    return idx