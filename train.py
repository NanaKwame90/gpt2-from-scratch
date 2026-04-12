"""
Supplementary code for the "Build a Large Language Model From Scratch" book
by Sebastian Raschka.

Book link: http://mng.bz/orYv
Code repository: https://github.com/rasbt/LLMs-from-scratch

© Sebastian Raschka. All rights reserved.
© Modifications for ANLP assignment: TA team ANLP2025

This code is provided for educational purposes and may be used, modified,
and distributed for non-commercial purposes, provided that proper attribution
to the book and author is included.
"""

import matplotlib.pyplot as plt                    # plotting utilities for loss curves
import os                                          # filesystem utilities
import torch                                       # tensor computations
import urllib.request                              # downloads for optional assets
import tiktoken                                    # GPT-2 tokenizer
import math                                        # math helpers
import json                                        # config loading

import argparse                                    # CLI parsing


from model.model_gpt import GPTModel               # core GPT model
from model.utils import create_dataloader_v1, generate_text_simple, decode_1, decode_2


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)                      # list of token IDs
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)   # add batch dimension (1, T)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)                               # drop batch dimension
    return tokenizer.decode(flat.tolist())                   # turn IDs back into text


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)                               # forward pass
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()          # CE over vocab
    )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches                     # average over sampled batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()                                           # disable dropout
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()                                          # switch back to train mode
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]           # max context length
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size,
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))          # avoid breaking console lines
    model.train()


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
    clipping=False,
    warmup_steps=0,
    max_grad_norm=1.0,
):
    """Train the model with optional gradient clipping and linear warmup."""

    # Persist the initial learning rate for warmup tracking
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])

    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()

            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()

            if clipping and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            # Linear warmup based on steps to avoid sudden LR spikes early on
            if warmup_steps > 0 and global_step < warmup_steps:
                warmup_scale = float(global_step + 1) / float(warmup_steps)
                for group in optimizer.param_groups:
                    group["lr"] = group["initial_lr"] * warmup_scale

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Ep {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )

        generate_and_print_sample(model, tokenizer, device, start_context)

    # Restore original learning rates after warmup
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"]

    return train_losses, val_losses, track_tokens_seen


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, show=False):
    """ 
    TODO: Implement a function that plots training and validation loss.

    Requirements:
    - Create a figure with matplotlib.
    - Plot train and validation loss against the number of epochs.
    - Add axis labels and a legend.
    - Add a second x-axis showing the number of tokens seen.
    - Save the figure as "loss-plot.png" and display it.

    Hints:
    - Use plt.subplots() to create a figure and an axis.
    - Use ax.plot(...) to plot lines.
    - Use ax.twiny() to create a second x-axis on top.
    - Use fig.tight_layout() before saving.
    - Use plt.savefig(...) and plt.show() at the end.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(epochs_seen, train_losses, label="Train loss")
    ax.plot(epochs_seen, val_losses, label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.legend()

    # Secondary x-axis to indicate tokens seen during training
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(ax.get_xticks())
    # Map epoch ticks to token counts using linear interpolation
    if len(tokens_seen) > 0:
        max_tokens = tokens_seen[-1]
        epoch_max = epochs_seen[-1] if len(epochs_seen) > 0 else 1
        mapped = [tick / epoch_max * max_tokens for tick in ax.get_xticks()]
        ax_top.set_xticklabels([f"{int(t):,}" for t in mapped])
        ax_top.set_xlabel("Tokens seen")

    fig.tight_layout()
    fig.savefig("loss-plot.png", dpi=150)
    if show:
        plt.show()

    return fig, ax
    

