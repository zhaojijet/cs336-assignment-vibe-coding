import logging
import argparse
import random
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from mlx_lm import load, generate
from mlx_lm.tuner.utils import linear_to_lora_layers

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.mlx_utils import (
    get_response_log_probs,
    masked_mean,
    masked_normalize,
    tokenize_prompt_and_output_mlx,
)

logger = logging.getLogger(__name__)


def compute_group_normalized_rewards(rewards):
    # rewards: list of floats
    # group normalize
    # (r - mean) / (std + eps)
    r = mx.array(rewards)
    mean = r.mean()
    std = r.std()
    return (r - mean) / (std + 1e-8)


def train_step(
    model,
    optimizer,
    input_ids,
    labels,
    response_mask,
    advantages,
    old_log_probs,
    cliprange,
):
    def loss_fn(model):
        logits = model(input_ids)
        log_probs = nn.log_softmax(logits, axis=-1)

        labels_for_gather = mx.where(labels == -100, mx.array(0), labels)
        labels_expanded = mx.expand_dims(labels_for_gather, axis=-1)
        gather_log_probs = mx.take_along_axis(
            log_probs, labels_expanded, axis=-1
        ).squeeze(-1)

        # Current policy log probs
        pi_log_probs = gather_log_probs

        # Ratio
        ratio = mx.exp(pi_log_probs - old_log_probs)

        # Surrogate 1: ratio * A
        surr1 = ratio * advantages

        # Surrogate 2: clip(ratio, 1-eps, 1+eps) * A
        surr2 = mx.clip(ratio, 1.0 - cliprange, 1.0 + cliprange) * advantages

        # Loss = -min(surr1, surr2)
        loss = -mx.minimum(surr1, surr2)

        # Mask
        # Average over valid response tokens
        loss = (loss * response_mask).sum() / (response_mask.sum() + 1e-9)

        return loss

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    loss_val, grads = loss_and_grad_fn(model)
    optimizer.update(model, grads)
    return loss_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="data/models/Qwen2.5-Math-1.5B",
    )
    parser.add_argument(
        "--dataset-path", type=str, default="data/MATH/validation.jsonl"
    )
    parser.add_argument("--output-dir", type=str, default="grpo_mlx_output")
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-iterations", type=int, default=10)
    parser.add_argument("--lora-layers", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--cliprange", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Load model
    logger.info(f"Loading model {args.model_path}")
    model, tokenizer = load(args.model_path)

    # Freeze everything
    model.freeze()

    # LoRA
    lora_config = {
        "rank": args.lora_rank,
        "alpha": args.lora_alpha,
        "scale": args.lora_alpha / args.lora_rank,
        "dropout": 0.05,
    }
    linear_to_lora_layers(model, args.lora_layers, lora_config)

    # Unfreeze LoRA adapters
    logger.info("Unfreezing LoRA adapters")
    unfrozen_count = 0
    for name, m in model.named_modules():
        if hasattr(m, "lora_a"):
            m.unfreeze()
            for attr in ["base", "linear"]:
                if hasattr(m, attr):
                    sub = getattr(m, attr)
                    if isinstance(sub, nn.Module):
                        sub.freeze()
            unfrozen_count += 1
    logger.info(f"Unfrozen {unfrozen_count} LoRA modules")

    optimizer = optim.AdamW(learning_rate=args.lr)

    # Prompts
    prompts = []
    gts = []
    with open(args.dataset_path, "r") as f:
        for line in f:
            item = json.loads(line)
            prompts.append(item.get("problem") or item.get("question"))
            gts.append(item.get("solution") or item.get("answer"))
            if args.limit > 0 and len(prompts) >= args.limit:
                break

    PROMPT_TEMPLATE = "User: {instruction}\nAssistant: <think>"

    state = [model.state, optimizer.state]

    logger.info("Starting GRPO training")
    for iteration in range(args.num_iterations):
        logger.info(f"Iteration {iteration}")

        # Shuffle prompts
        combined = list(zip(prompts, gts))
        random.shuffle(combined)
        prompts_shuffled, gts_shuffled = zip(*combined)

        for i in range(0, len(prompts_shuffled), args.batch_size):
            batch_prompts = prompts_shuffled[i : i + args.batch_size]
            batch_gts = gts_shuffled[i : i + args.batch_size]

            logger.info(f"Batch {i//args.batch_size}")

            # 1. Rollout: Generate Group
            all_responses = []
            all_full_prompts = []
            all_gts = []

            for prompt_text, gold_answer in zip(batch_prompts, batch_gts):
                full_prompt = PROMPT_TEMPLATE.format(instruction=prompt_text)
                for g in range(args.group_size):
                    res = generate(
                        model,
                        tokenizer,
                        prompt=full_prompt,
                        verbose=False,
                        max_tokens=args.max_new_tokens,
                    )
                    if res.startswith(full_prompt):
                        res = res[len(full_prompt) :].strip()
                    all_responses.append(res)
                    all_full_prompts.append(full_prompt)
                    all_gts.append(gold_answer)

            # 2. Compute Rewards and Advantages
            rewards = []
            for res, gold_answer in zip(all_responses, all_gts):
                r_meta = r1_zero_reward_fn(res, gold_answer)
                rewards.append(r_meta["reward"])

            advantages = compute_group_normalized_rewards(rewards)
            logger.info(f"Mean Reward: {mx.array(rewards).mean().item():.4f}")

            # 3. Training Update
            tokenized = tokenize_prompt_and_output_mlx(
                all_full_prompts, all_responses, tokenizer
            )

            input_ids = tokenized["input_ids"]
            labels = tokenized["labels"]
            response_mask = tokenized["response_mask"]

            # Get old log probs
            old_log_probs_dict = get_response_log_probs(model, input_ids, labels)
            old_log_probs = old_log_probs_dict["log_probs"]

            # Reshape advantages for broadcasting (total_rollouts, 1)
            adv_mx = advantages.reshape(-1, 1)

            loss = train_step(
                model,
                optimizer,
                input_ids,
                labels,
                response_mask,
                adv_mx,
                old_log_probs,
                args.cliprange,
            )
            mx.eval(state)

            logger.info(f"Step Loss: {loss.item():.4f}")

    # Save
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_weights(f"{args.output_dir}/adapters.npz")
    logger.info(f"Saved adapters to {args.output_dir}")


if __name__ == "__main__":
    main()
