import torch                              # tensor library used for computations
import torch.nn as nn                     # neural network modules and layers

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0               # ensure heads divide output evenly

        self.d_out = d_out                          # total projection size
        self.num_heads = num_heads                  # number of attention heads
        self.head_dim = d_out // num_heads          # dimension per head

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)  # linear map for queries
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)    # linear map for keys
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)  # linear map for values
        self.out_proj = nn.Linear(d_out, d_out)                # output projection after heads
        self.dropout = nn.Dropout(dropout)                     # dropout on attention weights
        # Upper-triangular causal mask (1s above diagonal) stored as buffer for device moves
        # Use a distinct name to avoid attribute conflicts with Module internals
        self.register_buffer('causal_mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

        
    def forward(self, x):
        b, num_tokens, d_in = x.shape                # (B, T, C)

        # Project tokens to queries, keys, values for all heads: (B, T, d_out)
        keys = self.W_key(x) # (B, T, d_out)
        queries = self.W_query(x) # (B, T, d_out)
        values = self.W_value(x) # (B, T, d_out)
        print(x.shape) # (B, T, C)
        
        # Reshape to (B, H, T, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores: (B, H, T, T)
        attn_scores = queries @ keys.transpose(2, 3)
        # slice the causal mask for the current sequence length and convert to a boolean mask on the same device
        mask_bool = self.causal_mask[:num_tokens, :num_tokens].to(attn_scores.device).bool()
        attn_scores.masked_fill_(mask_bool, float('-inf'))
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Context: (B, H, T, head_dim) -> (B, T, d_out)
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5                                # numerical stability constant
        self.scale = nn.Parameter(torch.ones(emb_dim))  # learnable gain
        self.shift = nn.Parameter(torch.zeros(emb_dim)) # learnable bias

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)            # mean over feature dimension
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # variance over features
        norm_x = (x - mean) / torch.sqrt(var + self.eps)    # normalize features
        return self.scale * norm_x + self.shift             # apply affine transform


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(             # Gaussian Error Linear Unit activation
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Position-wise MLP: expand to 4x then project back
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)                          # apply MLP to each token independently


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Masked multi-head self-attention sublayer
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        # Position-wise feed-forward network
        self.ff = FeedForward(cfg)
        # Pre-layer norms for attention and MLP paths
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        # Dropout applied to residual branches
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x                               # (B, T, C)
        x = self.norm1(x)                          # (B, T, C)
        x = self.att(x)                            # (B, T, C)
        x = self.drop_shortcut(x)
        x = x + shortcut                           # residual 1 keeps (B, T, C)

        shortcut = x                               # (B, T, C)
        x = self.norm2(x)                          # (B, T, C)
        x = self.ff(x)                             # (B, T, C)
        x = self.drop_shortcut(x)
        x = x + shortcut                           # residual 2 keeps (B, T, C)

        return x                                   # (B, T, C)


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Embedding tables: tokens and fixed positional indices
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # Stack of Transformer blocks (depth = n_layers)
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # Final layer norm before output head
        self.final_norm = LayerNorm(cfg["emb_dim"])
        # Linear head to map hidden states to vocabulary logits
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape           # (B, T)
        tok_embeds = self.tok_emb(in_idx)            # (B, T, C)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))  # (T, C)
        x = tok_embeds + pos_embeds                  # (B, T, C)
        x = self.drop_emb(x)                         # (B, T, C)
        x = self.trf_blocks(x)                       # (B, T, C)
        x = self.final_norm(x)                       # (B, T, C)
        logits = self.out_head(x)                    # (B, T, vocab_size)
        return logits                                # logits for next-token prediction
