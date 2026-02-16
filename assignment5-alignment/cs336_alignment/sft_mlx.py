import logging
import argparse
import random
import json
from pathlib import Path
import math
import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers

from cs336_alignment.sft import get_packed_sft_dataset
from cs336_alignment.mlx_utils import get_response_log_probs, masked_mean

logger = logging.getLogger(__name__)


def loss_fn(model, input_ids, labels, response_mask):
    logits = model(input_ids)
    log_probs = nn.log_softmax(logits, axis=-1)

    labels_for_gather = mx.where(labels == -100, mx.array(0), labels)
    labels_expanded = mx.expand_dims(labels_for_gather, axis=-1)
    gather_log_probs = mx.take_along_axis(log_probs, labels_expanded, axis=-1).squeeze(
        -1
    )

    token_loss = -gather_log_probs
    # Loss averaged over valid tokens in the micro-batch
    loss = (token_loss * response_mask).sum() / (response_mask.sum() + 1e-9)
    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", type=str, default="data/models/Qwen2.5-Math-1.5B"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/competition_math/data/train-00000-of-00001-7320a6f3aba8ebd2.parquet",
    )
    parser.add_argument("--output-dir", type=str, default="sft_mlx_output")
    parser.add_argument("--batch-size", type=int, default=4, help="Global batch size")
    parser.add_argument(
        "--micro-batch-size", type=int, default=1, help="Micro batch size per step"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Steps to accumulate gradients",
    )
    parser.add_argument("--lora-layers", type=int, default=16)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0, help="LoRA alpha")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Validation
    if args.batch_size % args.micro_batch_size != 0:
        raise ValueError("batch-size must be divisible by micro-batch-size")

    grad_accum_steps = args.batch_size // args.micro_batch_size
    logger.info(f"Global Batch Size: {args.batch_size}")
    logger.info(f"Micro Batch Size: {args.micro_batch_size}")
    logger.info(f"Gradient Accumulation Steps: {grad_accum_steps}")

    # Load model
    logger.info(f"Loading model {args.model_path}")
    model, tokenizer = load(args.model_path)

    # Convert to LoRA
    logger.info("Converting to LoRA")
    lora_config = {
        "rank": args.lora_rank,
        "alpha": args.lora_alpha,
        "scale": args.lora_alpha / args.lora_rank,
        "dropout": 0.05,
    }
    linear_to_lora_layers(model, args.lora_layers, lora_config)

    # Freeze everything first
    model.freeze()

    # Unfreeze only the LoRA adapters
    logger.info("Unfreezing LoRA adapters")
    unfrozen_count = 0
    for name, m in model.named_modules():
        if hasattr(m, "lora_a"):
            m.unfreeze()
            # Freeze the base layer which is usually 'linear' or 'base'
            for attr in ["linear", "base"]:
                if hasattr(m, attr):
                    sub = getattr(m, attr)
                    if isinstance(sub, nn.Module):
                        sub.freeze()
            unfrozen_count += 1

    if unfrozen_count == 0:
        logger.error("No LoRA layers found!")

    logger.info(f"Unfrozen {unfrozen_count} LoRA modules")

    # Optimizer
    optimizer = optim.AdamW(learning_rate=args.lr)

    # Dataset
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = get_packed_sft_dataset(
        args.dataset_path, tokenizer, seq_length=args.seq_length, limit=args.limit
    )

    if len(dataset) == 0:
        logger.warning("Dataset is empty!")
        return

    # State for checkpointing
    state = [model.state, optimizer.state]

    @mx.compile
    def step(batch_input_ids, batch_labels, batch_response_mask):
        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad_fn(
            model, batch_input_ids, batch_labels, batch_response_mask
        )
        return loss, grads

    logger.info("Starting training")

    steps_per_epoch = len(dataset) // args.batch_size
    global_step = 0

    for epoch in range(args.epochs):
        indices = list(range(len(dataset)))
        random.shuffle(indices)

        # We drop the last incomplete batch if any
        num_batches = len(indices) // args.batch_size

        for i in range(num_batches):
            batch_start = i * args.batch_size
            batch_indices = indices[batch_start : batch_start + args.batch_size]

            # Accumulate gradients
            accum_grads = None
            accum_loss = 0.0

            # Process micro-batches
            for j in range(0, len(batch_indices), args.micro_batch_size):
                micro_indices = batch_indices[j : j + args.micro_batch_size]
                micro_batch = [dataset[idx] for idx in micro_indices]

                # Collate
                # Stack to numpy first
                input_ids_np = np.array(
                    [item["input_ids"].numpy() for item in micro_batch]
                )
                labels_np = np.array([item["labels"].numpy() for item in micro_batch])
                mask_np = np.array(
                    [item["response_mask"].numpy() for item in micro_batch]
                )

                input_ids = mx.array(input_ids_np)
                labels = mx.array(labels_np)
                response_mask = mx.array(mask_np)

                loss, grads = step(input_ids, labels, response_mask)

                accum_loss += loss.item() * len(micro_indices)

                # Accumulate grads
                if accum_grads is None:
                    accum_grads = grads
                else:
                    accum_grads = tree_map(lambda x, y: x + y, accum_grads, grads)

            # Normalize grads
            accum_grads = tree_map(lambda x: x / grad_accum_steps, accum_grads)
            batch_loss = accum_loss / args.batch_size

            optimizer.update(model, accum_grads)
            mx.eval(state)

            global_step += 1
            if global_step % 1 == 0:  # Log every step since we run small test
                logger.info(
                    f"Epoch {epoch}, Step {global_step}, Loss: {batch_loss:.4f}"
                )

    # Save adapters
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_weights(f"{args.output_dir}/adapters.npz")
    logger.info(f"Saved adapters to {args.output_dir}")


if __name__ == "__main__":
    main()
