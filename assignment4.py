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

import matplotlib.pyplot as plt                 # for plotting losses
import os                                       # filesystem helpers
import torch                                    # tensor computations
import tiktoken                                 # GPT-2 tokenizer
import json                                     # config loading

import argparse                                 # CLI parsing
from collections import OrderedDict              # for state_dict key remapping

from model.model_gpt import GPTModel             # model definition
from model.utils import create_dataloader_v1, generate_text_simple, decode_1, decode_2
from pretrained import load_pretrained_gpt      # helper to load pretrained weights

from train import (
    text_to_token_ids,                          # text -> token IDs
    token_ids_to_text,                          # token IDs -> text
    train_model,                                # training loop
    plot_losses,                                # loss plotting
)


def calculate_perplexity(model, text, tokenizer, device):
    """Compute perplexity of a given text using the current model."""
    model.eval()

    tokens = tokenizer.encode(text)                             # tokenize input
    input_ids = torch.tensor(tokens).unsqueeze(0).to(device)    # shape (1, T)

    with torch.no_grad():
        logits = model(input_ids[:, :-1])                       # predict next tokens
        targets = input_ids[:, 1:]                              # shift by one
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
        )

    return torch.exp(loss).item()


def _remap_causal_mask_keys(state_dict):
    """Remap older checkpoints with 'causal_mask' buffer name to current 'mask'."""
    new_state = OrderedDict()
    for k, v in state_dict.items():
        if "att.causal_mask" in k:
            new_state[k.replace("att.causal_mask", "att.mask")] = v
        else:
            new_state[k] = v
    return new_state


def main(gpt_config, settings, train_text_path: str = "./data/frankenstein.txt"):
    torch.manual_seed(123)                                      # reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = tiktoken.get_encoding("gpt2")                   # GPT-2 BPE tokenizer

    with open(train_text_path, "r", encoding="utf-8") as file:
        text_data = file.read()                                 # load training corpus

    ##############################
    # Initialize model
    model = GPTModel(gpt_config)                                # init model
    model.to(device)
    optimizer = torch.optim.AdamW(                              # AdamW optimizer
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )

    ##############################
    # Set up dataloaders
    ##############################
    # Traing and Evalua
    train_ratio = 0.90                                          # 90/10 split
    split_idx = int(train_ratio * len(text_data))

    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],       # non-overlapping chunks
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )

    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

    ##############################
    # Train model
    ##############################

    train_losses, val_losses, tokens_seen = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=settings["num_epochs"],
        eval_freq=5,                                # evaluate every 5 steps
        eval_iter=1,                               # mini-eval batches
        start_context="My beloved Sister,",       # sample prefix for prints
        tokenizer=tokenizer,
        clipping=settings.get("clipping", False),
        warmup_steps=settings.get("warmup_steps", 0),
        max_grad_norm=settings.get("max_grad_norm", 1.0),
    )

    return train_losses, val_losses, tokens_seen, model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assignment 4 CLI")

    # ACTION: train, decode, perplexity, pretrained-model
    parser.add_argument(
        "action",
        choices=["train", "decode", "perplexity", "pretrained-model"],
        help="Which action to run.",
    )

    # Positional arguments whose meaning depends on action:
    #   train:          arg1 = optional train_text_path
    #   decode:         arg1 = model_number, arg2 = decoder_name, arg3 = optional start_text
    #   perplexity:     arg1 = model_number, arg2 = test_text_path
    #   pretrained-model: no extra args
    parser.add_argument("arg1", nargs="?", help="Meaning depends on action")
    parser.add_argument("arg2", nargs="?", help="Meaning depends on action")
    parser.add_argument("arg3", nargs="?", help="Meaning depends on action")

    args = parser.parse_args()

    with open("config.json", "r") as f:
        config = json.load(f)

    GPT_CONFIG = config["GPT_CONFIG"]
    OTHER_SETTINGS = config["OTHER_SETTINGS"]
    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ###################################################################
    # ACTION 1 — TRAIN
    ###################################################################
    if args.action == "train":
        # python assignment4.py train [train_text_path]
        #If arg1 is a digit, treat it as model number to train; else, it is a path.
        if args.arg1 and args.arg1.isdigit():
            idx = [int(args.arg1)]
            train_text_path = "./data/frankenstein.txt"
        else:
            idx = range(len(GPT_CONFIG))
            train_text_path = args.arg1 or "./data/frankenstein.txt"
        print(f"[TRAIN] Using training text file: {train_text_path}")   
        """
        TODO: Modify the config.json file to train different models.
        Keep in mind the computational cost of increasing or decreasing certain parameters.
        """
        for i in idx:
            print(f"[TRAIN] Starting training for model_{i}.pth...")
            train_losses, val_losses, tokens_seen, model = main(
                gpt_config=GPT_CONFIG[i],
                settings=OTHER_SETTINGS[i],
                train_text_path=train_text_path,
            )

            print(f"[TRAIN] Plotting losses for model_{i}.pth...")
            plot_losses(range(len(train_losses)), tokens_seen, train_losses, val_losses, show=False)
            
            # Save model checkpoint
            model_path = f"model_{i}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"[TRAIN] Saved trained model to {model_path}")
       

    ###################################################################
    # ACTION 2 — DECODE
    ###################################################################
    elif args.action == "decode":
        # python assignment4.py decode <model_number> <decoder_name> [start_text]
        if args.arg1 is None or args.arg2 is None:
            raise ValueError(
                "Usage: python assignment4.py decode <model_number> <decoder_name> [start_text]\n"
                "  decoder_name ∈ {decoder_default, decoder_1, decoder_2}"
            )

        try:
            model_num = int(args.arg1)
        except ValueError:
            raise ValueError("model_number must be an integer (e.g., 0, 1, 2).")


        decoder_name = args.arg2
        start_text = args.arg3 or "My beloved Sister,"

        model_path = f"model_{model_num}.pth"
        print(f"[DECODE] Using model file: {model_path}")
        print(f"[DECODE] Decoder: {decoder_name}")
        print(f"[DECODE] Start text: {start_text!r}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file '{model_path}' not found. Train first or choose another model_number."
            )

        model = GPTModel(GPT_CONFIG[model_num])               # rebuild model
        state = torch.load(model_path, map_location=device)   # load weights
        state = _remap_causal_mask_keys(state)
        model.load_state_dict(state)
        model.to(device)
        model.eval()

        encoded = text_to_token_ids(
            text=start_text,
            tokenizer=tokenizer,
        ).to(device)

        context_size = model.pos_emb.weight.shape[0]           # model context length

        with torch.no_grad():
            if decoder_name == "decoder_default":
                token_ids = generate_text_simple(
                    model=model,
                    idx=encoded,
                    max_new_tokens=50,
                    context_size=context_size,
                )
            elif decoder_name == "decoder_1":
                token_ids = decode_1(
                    model=model,
                    idx=encoded,
                    max_new_tokens=50,
                    context_size=context_size,
                )
            elif decoder_name == "decoder_2":
                token_ids = decode_2(
                    model=model,
                    idx=encoded,
                    max_new_tokens=50,
                    context_size=context_size,
                )
            else:
                raise ValueError(
                    "decoder_name must be one of: decoder_default, decoder_1, decoder_2."
                )

        decoded_text = token_ids_to_text(token_ids, tokenizer=tokenizer)
        print("\n[DECODE] Generated text:\n")
        print(decoded_text.replace("\n", " "))

    ###################################################################
    # ACTION 3 — PERPLEXITY
    ###################################################################
    elif args.action == "perplexity":
        # python assignment4.py perplexity <model_number> [test_text_path]
        if args.arg1 is None:
            raise ValueError(
                "Usage: python assignment4.py perplexity <model_number> [test_text_path]"
            )

        try:
            model_num = int(args.arg1)
        except ValueError:
            raise ValueError("model_number must be an integer (e.g., 0, 1, 2).")

        test_text_path = args.arg2 or "data/die_automata.txt"

        print(f"[PERPLEXITY] Using model_{model_num}.pth on text file: {test_text_path}")

        model_path = f"model_{model_num}.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file '{model_path}' not found. Train first or choose another model_number."
            )

        if not os.path.exists(test_text_path):
            raise FileNotFoundError(
                f"Text file '{test_text_path}' not found. Provide a valid path."
            )

        model = GPTModel(GPT_CONFIG[model_num])               # rebuild model
        state = torch.load(model_path, map_location=device)   # load weights
        state = _remap_causal_mask_keys(state)
        model.load_state_dict(state)
        model.to(device)
        model.eval()

        with open(test_text_path, "r", encoding="utf-8") as file:
            test_text = file.read()                           # read evaluation corpus

        token_ids = text_to_token_ids(text=test_text, tokenizer=tokenizer).to(device)
        if token_ids.numel() < 2:
            raise ValueError("Test text is too short to compute perplexity.")

        context_size = model.pos_emb.weight.shape[0]           # model context length

        total_log_likelihood = 0.0                            # accumulate log p(x_i)
        total_tokens = 0

        with torch.no_grad():
            for start in range(0, token_ids.shape[1] - 1, context_size):
                input_chunk = token_ids[:, start : start + context_size]
                target_chunk = token_ids[:, start + 1 : start + context_size + 1]

                logits = model(input_chunk)                    # model predictions
                log_probs = torch.log_softmax(logits, dim=-1)  # convert to log-probs

                seq_len = target_chunk.shape[1]
                log_probs_next = log_probs[:, :seq_len, :].gather(
                    -1, target_chunk[:, :seq_len].unsqueeze(-1)
                ).squeeze(-1)                                  # log p(x_i | context)

                total_log_likelihood += log_probs_next.sum().item()
                total_tokens += seq_len

        avg_nll = -total_log_likelihood / total_tokens        # average NLL
        perplexity = torch.exp(torch.tensor(avg_nll))          # exp of NLL = perplexity

        print(f"[PERPLEXITY] Perplexity on '{test_text_path}': {perplexity.item():.4f}")

    ###################################################################
    # ACTION 4 — PRETRAINED MODEL (BONUS)
    ###################################################################
    elif args.action == "pretrained-model":
        # python assignment4.py pretrained-model
        print("[PRETRAINED] Loading pretrained GPT model...")
        start_context = "My beloved Sister,"
        load_pretrained_gpt(start_context=start_context)
