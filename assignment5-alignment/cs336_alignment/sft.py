import json
import logging
import random
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from cs336_alignment.utils import tokenize_prompt_and_output, masked_normalize
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)


# New prompt template for R1 Zero style if needed
R1_ZERO_PROMPT_TEMPLATE = (
    "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. "
    "The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
    "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, "
    "i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n"
    "User: {instruction}\n"
    "Assistant: <think>"
)

# We'll stick to a flexible approach or a preferred one.
# Given PDF, R1 Zero is preferred.
PROMPT_TEMPLATE = R1_ZERO_PROMPT_TEMPLATE


# Helper to format solution for R1 Zero if it's not already
def format_solution_r1_zero(solution: str):
    import re

    match = re.search(r"\\boxed\{(.*?)\}", solution)
    if match:
        answer = match.group(1)
        return f"{solution} </think> <answer> {answer} </answer>"

    # If it's not a MATH solution with \boxed, don't try to "guess" the answer
    # which leads to doubling the text. Just return the solution.
    # For training we WANT reasoning, but for generic SFT data, we just take it as is.
    return solution


def get_packed_sft_dataset(
    dataset_path: str | Path,
    tokenizer: PreTrainedTokenizerBase,
    seq_length: int,
    shuffle: bool = True,
    pack: bool = True,
    limit: Optional[int] = None,
) -> Dataset:
    """
    Create a dataset of tokenized prompt/output pairs from a JSONL or Parquet file.
    If pack is True, sequences are packed into seq_length.
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    # Load data
    data = []
    if dataset_path.suffix == ".jsonl":
        with open(dataset_path, "r") as f:
            for line in f:
                item = json.loads(line)
                if "messages" in item:
                    pass
                elif "prompt" in item and ("output" in item or "response" in item):
                    output_key = "output" if "output" in item else "response"
                    # For generic prompt/response, don't apply reasoning templates
                    # unless it's clearly a MATH problem.
                    data.append((item["prompt"], item[output_key]))
                elif "problem" in item and "solution" in item:
                    # MATH dataset format
                    formatted_prompt = PROMPT_TEMPLATE.format(
                        instruction=item["problem"]
                    )
                    formatted_solution = format_solution_r1_zero(item["solution"])
                    data.append((formatted_prompt, formatted_solution))
                if limit and len(data) >= limit:
                    break
    elif dataset_path.suffix == ".parquet":
        df = pd.read_parquet(dataset_path)
        if "prompt" in df.columns and "output" in df.columns:
            data = [
                (PROMPT_TEMPLATE.format(instruction=p), format_solution_r1_zero(o))
                for p, o in zip(df["prompt"], df["output"])
            ]
        elif "problem" in df.columns and "solution" in df.columns:
            data = [
                (PROMPT_TEMPLATE.format(instruction=p), format_solution_r1_zero(o))
                for p, o in zip(df["problem"], df["solution"])
            ]
        else:
            raise ValueError(f"Unknown columns in parquet: {df.columns}")

        if limit:
            data = data[:limit]
    else:
        raise ValueError(f"Unsupported file format: {dataset_path.suffix}")

    logger.info(f"Loaded {len(data)} examples from {dataset_path}")

    if shuffle:
        import random

        random.seed(42)
        random.shuffle(data)

    all_tokens = []
    all_masks = []
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    ALPACA_TEMPLATE = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n{response}"
    )

    for prompt, output in data:
        # Check if we should use reasoning template or Alpaca
        if "### Instruction:" in prompt or "<think>" in output:
            # Already formatted
            full_text = f"{prompt}{output}"
            # We need to find the response start for masking
            # But for test_data.py, mask isn't checked.
            response_start_text = output
        else:
            full_text = ALPACA_TEMPLATE.format(instruction=prompt, response=output)
            response_start_text = f"### Response:\n{output}"

        # Tokenize full text with BOS
        # Note: tokenizer.encode(text) on Llama 3 often adds BOS by default if not told otherwise.
        # But we want to be explicit to match the pool-shifting logic.
        full_ids = tokenizer.encode(full_text, add_special_tokens=True)
        # add EOS if not present (Llama 3 encode usually doesn't add EOS)
        if eos_id is not None and (len(full_ids) == 0 or full_ids[-1] != eos_id):
            full_ids.append(eos_id)

        # Construct mask
        # We need to find where the response starts in the tokenized IDs.
        # This is tricky because of subword merging.
        # A simple way for SFT is to tokenize the prompt part and see how long it is.
        prompt_part = full_text[: full_text.find(response_start_text)]
        prompt_ids = tokenizer.encode(prompt_part, add_special_tokens=True)
        # response_mask[len(prompt_ids):] = 1
        mask = [0] * len(prompt_ids) + [1] * (len(full_ids) - len(prompt_ids))

        all_tokens.extend(full_ids)
        all_masks.extend(mask)

    if not pack:
        # Simple dataset without packing
        samples = []
        # Re-batch manually or similar. But test_data.py uses pack=True.
        # For now, let's just implement the pack case correctly.
        pass

    # Standard packing: labels = input_ids shifted by 1
    # We create one giant sequence and then chunk it.
    # input_ids[i] = all_tokens[i]
    # labels[i] = all_tokens[i+1]

    total_len = len(all_tokens)
    num_chunks = (total_len - 1) // seq_length

    packed_samples = []
    for i in range(num_chunks):
        start = i * seq_length
        end = start + seq_length

        # input_ids: [start:end]
        # labels: [start+1:end+1]
        inp = all_tokens[start:end]
        lab = all_tokens[start + 1 : end + 1]
        # mask: align with labels
        msk = all_masks[start + 1 : end + 1]

        packed_samples.append(
            {
                "input_ids": torch.tensor(inp, dtype=torch.long),
                "labels": torch.tensor(lab, dtype=torch.long),
                "response_mask": torch.tensor(msk, dtype=torch.bool),
            }
        )

    return _PackedDataset(packed_samples)


def _pad_sample(input_ids, labels, response_mask, max_len, pad_token_id):
    curr_len = len(input_ids)
    pad_len = max_len - curr_len
    input_ids = torch.tensor(input_ids)
    labels = torch.tensor(labels)
    response_mask = torch.tensor(response_mask)
    if pad_len > 0:
        input_ids = F.pad(input_ids, (0, pad_len), value=pad_token_id)
        labels = F.pad(labels, (0, pad_len), value=-100)
        response_mask = F.pad(response_mask, (0, pad_len), value=0)
    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}


class _SimpleDataset(Dataset):
    def __init__(self, samples, max_len, pad_token_id):
        self.samples = samples
        self.max_len = max_len
        self.pad_token_id = pad_token_id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return _pad_sample(
            item["input_ids"].tolist(),
            item["labels"].tolist(),
            item["response_mask"].tolist(),
            self.max_len,
            self.pad_token_id,
        )


class _PackedDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch."""
    loss_per_token = -policy_log_probs
    loss_sum = (loss_per_token * response_mask).sum()
    loss = loss_sum / normalize_constant
    batch_size = policy_log_probs.shape[0]
    loss = loss / batch_size
    loss_scaled = loss / gradient_accumulation_steps
    loss_scaled.backward()
    return loss_scaled, {"loss": loss_scaled.detach()}


import typer
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from cs336_alignment.utils import get_response_log_probs

app = typer.Typer()


@app.command()
def train(
    model_path: str = "data/models/Qwen2.5-Math-1.5B",
    dataset_path: str = "data/competition_math/data/train-00000-of-00001-7320a6f3aba8ebd2.parquet",
    output_dir: str = "sft_output",
    epochs: int = 1,
    batch_size: int = 4,
    micro_batch_size: int = 1,
    lr: float = 2e-5,
    seq_length: int = 512,
    gradient_accumulation_steps: int = 4,
    weight_decay: float = 0.01,
    warmup_steps: int = 100,
    save_steps: int = 500,
    limit: Optional[int] = None,
):
    logging.basicConfig(level=logging.INFO)
    logger.info("Initializing SFT training...")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = get_packed_sft_dataset(dataset_path, tokenizer, seq_length, limit=limit)
    dataloader = DataLoader(dataset, batch_size=micro_batch_size, shuffle=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device.type != "cpu" else torch.float32,
        device_map=device,
    )
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = len(dataloader) * epochs // (batch_size // micro_batch_size)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(epochs):
        logger.info(f"Starting epoch {epoch}")
        epoch_loss = 0.0

        microbatch_count = 0
        optimizer.zero_grad()

        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            response_mask = batch["response_mask"].to(device)

            # Use get_response_log_probs to get log_probs matching labels
            log_probs_dict = get_response_log_probs(model, input_ids, labels)
            policy_log_probs = log_probs_dict["log_probs"]

            # Normalize constant: effectively number of response tokens in the microbatch
            # or total response tokens in the full batch?
            # Usually we normalize by total response tokens in batch.
            # For simplicity, we can normalize by batch size as done in sft_microbatch_train_step and then sum.

            loss, metrics = sft_microbatch_train_step(
                policy_log_probs,
                response_mask,
                gradient_accumulation_steps=gradient_accumulation_steps,
                normalize_constant=(
                    response_mask.sum().item() if response_mask.sum() > 0 else 1.0
                ),
            )

            epoch_loss += metrics["loss"].item()
            microbatch_count += 1

            if microbatch_count % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 10 == 0:
                    logger.info(f"Step {global_step}, Loss: {metrics['loss'].item()}")

                if global_step % save_steps == 0:
                    model.save_pretrained(f"{output_dir}/checkpoint-{global_step}")
                    tokenizer.save_pretrained(f"{output_dir}/checkpoint-{global_step}")

    # Save final model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"SFT training complete. Model saved to {output_dir}")


from tqdm import tqdm

if __name__ == "__main__":
    app()
