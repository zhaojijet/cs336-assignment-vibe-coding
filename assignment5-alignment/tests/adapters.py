from __future__ import annotations

import os
from typing import Any, Callable, Literal

import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


from cs336_alignment.utils import (
    tokenize_prompt_and_output,
    compute_entropy,
    get_response_log_probs,
    masked_mean,
    masked_normalize,
)
from cs336_alignment.rl_utils import (
    compute_naive_policy_gradient_loss,
    compute_group_normalized_rewards,
    compute_grpo_clip_loss,
    compute_policy_gradient_loss,
)


def run_tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """
    Tokenize the prompt and output strings, and construct a mask that is 1
    for the output tokens and 0 for the prompt tokens.

    Args:
        prompt_strs: list[str]
            List of prompt strings.
        output_strs: list[str]
            List of output strings.
        tokenizer: PreTrainedTokenizerBase
            Tokenizer to use.

    Returns:
        dict[str, torch.Tensor]: A dictionary with the following keys:
            - "input_ids": Tensor of shape (batch_size, sequence_length) with the
              token IDs for the prompt and output.
            - "labels": Tensor of shape (batch_size, sequence_length) with the
              token IDs for the prompt and output.
            - "response_mask": Tensor of shape (batch_size, sequence_length) with 1s
              for the output tokens and 0s for the prompt tokens.
    """
    return tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer, target_len=9)


def run_compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute rewards for each group of rollout responses,
    optionally normalizing them within each group.

    For more on GRPO, see:
    https://arxiv.org/pdf/2402.03300.pdf
    DeepSeek-R1: https://arxiv.org/abs/2501.12948

    Args:
        reward_fn: Callable[[str, str], dict[str, float]]
            Function that takes a response and a ground truth and returns a dictionary
            of rewards.
        rollout_responses: list[str]
            List of rollout responses.
        repeated_ground_truths: list[str]
            List of ground truths, repeated for each response.
        group_size: int
            Size of each group (G).
        advantage_eps: float
            Epsilon to add to the standard deviation for numerical stability.
        normalize_by_std: bool
            Whether to normalize the rewards by the standard deviation.

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
            - advantages: torch.Tensor of shape (rollout_batch_size,): with the computed advantages.
            - raw_rewards: Tensor of shape (batch_size, 1) with the raw rewards.
            - metadata: Dictionary with metadata about the rewards.
    """
    advantages, raw_rewards, metadata = compute_group_normalized_rewards(
        reward_fn,
        rollout_responses,
        repeated_ground_truths,
        group_size,
        advantage_eps,
        normalize_by_std,
    )
    return advantages, raw_rewards, metadata


def run_compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the logits (i.e., entropy of the final dimension)."""
    return compute_entropy(logits)


def run_get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> dict[str, torch.Tensor]:
    """
    Get the log probabilities of the response tokens.

    Args:
        model: PreTrainedModel, the model to score.
        input_ids: torch.Tensor of shape (batch_size, sequence_length):
            Tensor of token IDs.
        labels: torch.Tensor of shape (batch_size, sequence_length):
            Tensor of labels.
        return_token_entropy: bool
            Whether to return the entropy of the token distribution.

    Returns:
        dict[str, torch.Tensor]: A dictionary with the following keys:
            - "log_probs": torch.Tensor of shape (batch_size, sequence_length):
              The log probabilities of the response tokens.
            - "token_entropy": torch.Tensor of shape (batch_size, sequence_length):
              The entropy of the token distribution (if return_token_entropy is True).
            - "logits": torch.Tensor of shape (batch_size, sequence_length, vocab_size):
              The logits of the response tokens.
    """
    model.eval()
    # Snapshots might have been generated in float32.
    # Convert to float32 and ensure CPU for matching.
    model.float()
    with torch.no_grad():
        return get_response_log_probs(model, input_ids, labels, return_token_entropy)


def run_compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the naive policy gradient loss.

    Args:
        raw_rewards_or_advantages: torch.Tensor of shape (batch_size, 1):
             The rewards or advantages for each example.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length):
             The log probabilities of the policy for each token.

    Returns:
        torch.Tensor of shape (batch_size, sequence_length):
             the policy gradient per-token loss.
    """
    return compute_naive_policy_gradient_loss(
        raw_rewards_or_advantages, policy_log_probs
    )


def run_compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the GRPO-Clip loss.

    Args:
        advantages: torch.Tensor of shape (batch_size, 1):
            the advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length):
            the log-probs of the policy.
        old_log_probs: torch.Tensor of shape (batch_size, sequence_length):
            the log-probs of the old policy.
        cliprange: float
            the clipping range for the PPO-style objective.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            - torch.Tensor of shape (batch_size, sequence_length):
                the GRPO-Clip per-token loss.
            - dict[str, torch.Tensor]:
                metadata about the loss.
    """
    return compute_grpo_clip_loss(
        advantages, policy_log_probs, old_log_probs, cliprange
    )


def run_compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Wrapper that delegates to the appropriate policy gradient loss function above.
    """
    return compute_policy_gradient_loss(
        policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange
    )


def run_masked_mean(
    tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None
) -> torch.Tensor:
    """Compute the mean of the tensor along a dimension,
    ignoring masked elements.

    Args:
        tensor: torch.Tensor, the tensor to compute the mean of.
        mask: torch.Tensor, the mask (1 for valid elements, 0 for ignored elements).
        dim: int, the dimension to compute the mean along.

    Returns:
        torch.Tensor, the mean of the tensor along the specified
        dimension, ignoring masked elements.
    """
    return masked_mean(tensor, mask, dim)


def run_sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length):
            the log-probs of the policy.
        response_mask: torch.Tensor of shape (batch_size, sequence_length):
            the mask for the response.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            - the policy gradient loss and its metadata.
    """

    return _sft_microbatch_train_step(
        policy_log_probs,
        response_mask,
        gradient_accumulation_steps,
        normalize_constant,
    )


def run_grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    pg_loss_per_token, metadata = compute_policy_gradient_loss(
        policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange
    )

    # Aggregate loss.
    # 1. masked_mean per example (dim=1) -> (B,)
    per_example_loss = masked_mean(pg_loss_per_token, response_mask, dim=1)

    # 2. Average over batch dimension -> scalar
    # Usually just mean().
    total_loss = per_example_loss.mean()

    # 3. Backward
    total_loss_scaled = total_loss / gradient_accumulation_steps
    total_loss_scaled.backward()

    return total_loss_scaled, metadata


def run_masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant,
    ignoring masked elements.

    Args:
        tensor: torch.Tensor, the tensor to sum and normalize.
        mask: torch.Tensor, the mask (1 for valid elements, 0 for ignored elements).
        dim: int, the dimension to sum along.
        normalize_constant: float, the constant to normalize by.

    Returns:
        torch.Tensor, the normalized sum, where masked elements
        are ignored.
    """
    return masked_normalize(tensor, mask, normalize_constant, dim)


"""
The below adapters are used in the optional 
RLHF / safety part of the Alignment assignment.
"""


from cs336_alignment.sft import (
    get_packed_sft_dataset as _get_packed_sft_dataset,
    sft_microbatch_train_step as _sft_microbatch_train_step,
)

from torch.utils.data import DataLoader


def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    """
    Given a tokenizer and a path to a dataset with instruction-tuning examples,
    construct a PyTorch Dataset for language modeling. The examples should be
    packed, i.e., all sequences in the dataset are of a constant length (`seq_length`).

    Args:
        tokenizer: transformers.PreTrainedTokenizerBase
            Transformers tokenizer to use in tokenizing and encoding text.
        dataset_path: str
            Path to file with instruction-tuning examples.
        seq_length: int
            Number of tokens to include in each example.
        shuffle: bool
            If true, shuffle the documents before packing them into examples.

    Returns:
        PyTorch Dataset for language modeling. Each example in this dataset is a dictionary of
        with keys "input_ids" and "labels" (both tensors of shape (seq_length, )).
        "input_ids" contains the token IDs for the language modeling inputs, and "labels" contains
        the token IDs for the language modeling labels.
    """
    return _get_packed_sft_dataset(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        seq_length=seq_length,
        shuffle=shuffle,
        pack=True,
    )


def run_iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    """
    Given a PyTorch Dataset, return an iterable over batches of size `batch_size`.
    Iterating through the returned iterable should constitute one epoch over the Dataset.

    Args:
        dataset: Dataset
            Dataset to emit batches from.
        batch_size: int
            Number of examples to include per batch.
        shuffle: bool
            If true, shuffle examples before batching them.

    Returns:
        Iterable over batches, where each batch has size `batch_size`.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        # Default collate_fn handles stacking tensors if they are same size
    )


def run_parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """
    Given an MMLU example and a model output, parse the model output into a
    predicted option letter (i.e., 'A', 'B', 'C', or 'D'). If the model output
    cannot be parsed into a prediction option letter, return None.

    mmlu_example: dict[str, Any]
        Dictionary with an MMLU example. Contains the following keys:
        - "subject": str with the subject of the question.
        - "question": str with the text of the question.
        - "options": list[str] with the four answer options (in order).
                     The first option refers to letter "A", the second to "B", etc.
        - "answer": str with the option of the correct answer (e.g., "A")
    model_output: str
        str with the model's output to the MMLU example.

    Returns:
        str (one of "A", "B", "C", or "D") if the model output can be parsed into a prediction,
        else None.
    """
    from cs336_alignment.metrics import parse_mmlu_response

    return parse_mmlu_response(mmlu_example, model_output)


def run_parse_gsm8k_response(
    model_output: str,
) -> str | None:
    """
    Given a GSM8K model output, parse the model output into a predicted numeric answer by
    taking the last number that occurs in the output.

    model_output: str
        str with the model's output to a GSM8K example.

    Returns:
        str with the predicted numeric answer if the model output can be parsed into a prediction,
        else None.
    """
    from cs336_alignment.metrics import parse_gsm8k_response

    return parse_gsm8k_response(model_output)


def run_compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """
    Given two language models (`lm`, and the "reference model" `lm_ref`),
    their tokenizer, the DPO beta hyperparameter, a prompt and a pair
    of responses to the prompt, computes the value of the DPO loss for this example.

    lm: torch.nn.Module
        Language model being trained.
    lm_ref: torch.nn.Module
        Reference language model.
    tokenizer: PreTrainedTokenizerBase
        Tokenizer for both language models.
    beta: float
        DPO beta hyperparameter.
    prompt: str
        Prompt for this instance of preference pair.
    response_chosen: str
        Preferred response to the prompt.
    response_rejected: str
        Rejected response to the prompt.

    Returns:
        torch.Tensor with the DPO loss for this example.
    """
    from cs336_alignment.dpo import compute_per_instance_dpo_loss

    return compute_per_instance_dpo_loss(
        lm,
        lm_ref,
        tokenizer,
        beta,
        prompt,
        response_chosen,
        response_rejected,
        target_len=9,
    )
