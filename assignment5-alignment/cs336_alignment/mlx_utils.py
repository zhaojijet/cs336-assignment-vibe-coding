import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Dict


def get_response_log_probs(
    model: nn.Module,
    input_ids: mx.array,
    labels: mx.array,
) -> Dict[str, mx.array]:
    """
    Get the conditional log-probs of the response given the prompt
    (where labels != -100).
    The model should return logits.

    Args:
        model: MLX model (returns logits).
        input_ids: mx.array of shape (batch_size, seq_len)
        labels: mx.array of shape (batch_size, seq_len)
            shifted input_ids. Tokens to ignore (prompt/padding) are -100.

    Returns:
        dict with "log_probs": mx.array (batch_size, seq_len)
    """
    # Forward pass
    logits = model(input_ids)

    # Log softmax
    log_probs = nn.log_softmax(logits, axis=-1)

    # Gather
    # labels has -100. We need to mask that or clamp for gather.
    # In MLX, gather/take is slightly different.
    # We can use take_along_axis.

    # Mask out -100 for indexing
    labels_for_gather = mx.where(labels == -100, mx.array(0), labels)
    # Expand labels for gather: (batch, seq, 1)
    labels_expanded = mx.expand_dims(labels_for_gather, axis=-1)

    # Gather log probs for the target token
    # log_probs: (batch, seq, vocab)
    gathered_log_probs = mx.take_along_axis(log_probs, labels_expanded, axis=-1)
    gathered_log_probs = mx.squeeze(gathered_log_probs, axis=-1)

    # Mask out ignored tokens in the output
    mask = labels != -100
    gathered_log_probs = mx.where(mask, gathered_log_probs, mx.array(0.0))

    return {"log_probs": gathered_log_probs}


def tokenize_prompt_and_output_mlx(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer,
    target_len: int | None = None,
) -> Dict[str, mx.array]:
    """Tokenize prompt and output, create response mask (MLX version)."""
    assert len(prompt_strs) == len(output_strs)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    batch_size = len(prompt_strs)
    input_ids_list = []
    labels_list = []
    mask_list = []

    p_ids_batch = [tokenizer.encode(p, add_special_tokens=False) for p in prompt_strs]
    o_ids_batch = [tokenizer.encode(o, add_special_tokens=False) for o in output_strs]
    eos_id = tokenizer.eos_token_id

    if target_len is None:
        lengths = [len(p) + len(o) + 1 for p, o in zip(p_ids_batch, o_ids_batch)]
        actual_target_len = max(lengths) if lengths else 0
    else:
        actual_target_len = target_len

    for i in range(batch_size):
        p_ids = p_ids_batch[i]
        o_ids = o_ids_batch[i]
        full_ids = p_ids + o_ids + [eos_id]

        # input_ids: full_ids[:-1]
        # labels: full_ids[1:]
        # response_mask: mask for o_ids + eos_id

        curr_input_ids = full_ids[:-1]
        curr_labels = full_ids[1:]

        curr_mask = [False] * len(curr_labels)
        start_mask = len(p_ids) - 1
        end_mask = len(p_ids) + len(o_ids)

        start_mask = max(0, start_mask)
        for idx in range(start_mask, min(end_mask, len(curr_mask))):
            curr_mask[idx] = True

        # Padding
        if len(curr_input_ids) < actual_target_len - 1:
            pad_len = (actual_target_len - 1) - len(curr_input_ids)
            curr_input_ids += [tokenizer.pad_token_id] * pad_len
            curr_labels += [-100] * pad_len
            curr_mask += [False] * pad_len
        elif len(curr_input_ids) > actual_target_len - 1:
            curr_input_ids = curr_input_ids[: actual_target_len - 1]
            curr_labels = curr_labels[: actual_target_len - 1]
            curr_mask = curr_mask[: actual_target_len - 1]

        input_ids_list.append(curr_input_ids)
        labels_list.append(curr_labels)
        mask_list.append(curr_mask)

    return {
        "input_ids": mx.array(input_ids_list),
        "labels": mx.array(labels_list),
        "response_mask": mx.array(mask_list),
    }


def masked_mean(
    tensor: mx.array, mask: mx.array, axis: Optional[int] = None
) -> mx.array:
    """
    Compute mean of tensor elements where mask is True.
    """
    if axis is None:
        return (tensor * mask).sum() / mask.sum()
    else:
        return (tensor * mask).sum(axis=axis) / mask.sum(axis=axis)


def masked_normalize(
    tensor: mx.array,
    mask: mx.array,
    normalize_constant: float = 1.0,
    axis: Optional[int] = None,
) -> mx.array:
    """
    Sum tensor elements where mask is True and divide by constant.
    """
    if axis is None:
        return (tensor * mask).sum() / normalize_constant
    else:
        return (tensor * mask).sum(axis=axis) / normalize_constant
