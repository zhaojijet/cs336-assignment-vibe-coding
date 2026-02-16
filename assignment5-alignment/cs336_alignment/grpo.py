import logging
import json
import random
import torch
import torch.nn.functional as F
from typing import Literal, List, Dict, Any, Optional
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from torch.utils.data import DataLoader
import typer
from tqdm import tqdm

from cs336_alignment.utils import (
    get_response_log_probs,
    masked_mean,
    tokenize_prompt_and_output,
)
from cs336_alignment.rl_utils import (
    compute_group_normalized_rewards,
    compute_policy_gradient_loss,
)
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

logger = logging.getLogger(__name__)

app = typer.Typer()


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute the GRPO/PG loss and backprop its gradients for a microbatch."""

    pg_loss_per_token, metadata = compute_policy_gradient_loss(
        policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange
    )

    # Aggregate loss.
    # 1. masked_mean per example (dim=1) -> (B,)
    # response_mask has shape (B, L)
    per_example_loss = masked_mean(pg_loss_per_token, response_mask, dim=1)

    # 2. Average over batch dimension -> scalar
    total_loss = per_example_loss.mean()

    # 3. Backward
    total_loss_scaled = total_loss / gradient_accumulation_steps
    total_loss_scaled.backward()

    return total_loss_scaled, metadata


@app.command()
def train(
    model_path: str = "data/models/Qwen2.5-Math-1.5B",
    dataset_path: str = "data/MATH/validation.jsonl",
    output_dir: str = "grpo_output",
    num_iterations: int = 10,
    group_size: int = 4,
    batch_size: int = 2,
    micro_batch_size: int = 1,
    lr: float = 1e-6,
    max_new_tokens: int = 512,
    cliprange: float = 0.2,
    advantage_eps: float = 1e-8,
    normalize_by_std: bool = True,
    limit: int = 10,
):
    logging.basicConfig(level=logging.INFO)
    logger.info("Initializing GRPO training...")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device.type != "cpu" else torch.float32,
        device_map=device,
    )

    # We might need a reference model if we want KL,
    # but based on my rl_utils.py implementation, we focus on clipping and advantages.

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Load prompt template
    with open("cs336_alignment/prompts/r1_zero.prompt", "r") as f:
        prompt_template = f.read()

    # Load small subset for on-policy training
    prompts = []
    ground_truths = []
    with open(dataset_path, "r") as f:
        for line in f:
            ex = json.loads(line)
            question = ex.get("problem") or ex.get("question")
            answer = ex.get("solution") or ex.get("answer")
            prompts.append(prompt_template.format(question=question))
            ground_truths.append(answer)
            if limit > 0 and len(prompts) >= limit:
                break

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for iteration in range(num_iterations):
        logger.info(f"Iteration {iteration}")

        # 1. Rollout: Sample G responses per prompt
        # In a real setup we'd use a DataLoader to shuffle prompts
        # For this demo/assignment, we'll just process them in batches

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            batch_gts = ground_truths[i : i + batch_size]

            # For each prompt in batch, generate G samples
            # Flatten rollout_responses to (batch_size * group_size)
            rollout_responses = []
            repeated_gts = []

            model.eval()
            with torch.no_grad():
                # We can batch generate if tokenizer/model supports it
                inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(
                    device
                )
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    num_return_sequences=group_size,
                    temperature=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

                # Decode
                decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

                # Extract Assistant responses
                for idx, prompt_text in enumerate(batch_prompts):
                    for g in range(group_size):
                        full_sample = decoded[idx * group_size + g]
                        # Assume prompt is at the beginning
                        response = full_sample[len(prompt_text) :].strip()
                        rollout_responses.append(response)
                        repeated_gts.append(batch_gts[idx])

            # 2. Reward and Advantages
            advantages, raw_rewards, metadata = compute_group_normalized_rewards(
                r1_zero_reward_fn,
                rollout_responses,
                repeated_gts,
                group_size=group_size,
                advantage_eps=advantage_eps,
                normalize_by_std=normalize_by_std,
            )

            logger.info(
                f"Batch {i//batch_size}, Mean Reward: {metadata['mean_reward']:.4f}"
            )

            # 3. Training Step
            model.train()
            optimizer.zero_grad()

            # Prepare inputs for SFT-like backward
            # We need input_ids, labels, response_mask for the rollout trajectories
            # And also old_log_probs

            # Tokenize rollouts
            # Trajectories = prompt + response
            tokenized_rollouts = tokenize_prompt_and_output(
                [prompt for prompt in batch_prompts for _ in range(group_size)],
                rollout_responses,
                tokenizer,
                target_len=None,  # Use actual length
            )

            input_ids = tokenized_rollouts["input_ids"].to(device)
            labels = tokenized_rollouts["labels"].to(device)
            response_mask = tokenized_rollouts["response_mask"].to(device)

            # Get old log probs before update
            with torch.no_grad():
                old_log_probs_dict = get_response_log_probs(model, input_ids, labels)
                old_log_probs = old_log_probs_dict["log_probs"]

            # Compute current log probs
            log_probs_dict = get_response_log_probs(model, input_ids, labels)
            policy_log_probs = log_probs_dict["log_probs"]

            # Advantages shape: (B*G,)
            # Reshape to (B*G, 1) for broadcasting in loss
            adv_tensor = advantages.to(device).unsqueeze(-1)

            loss, loss_meta = grpo_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=1,
                loss_type="grpo_clip",
                advantages=adv_tensor,
                old_log_probs=old_log_probs,
                cliprange=cliprange,
            )

            optimizer.step()

        # Save checkpoint
        if (iteration + 1) % 5 == 0:
            model.save_pretrained(f"{output_dir}/checkpoint-{iteration+1}")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    app()
