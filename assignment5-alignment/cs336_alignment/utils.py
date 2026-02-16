import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
    target_len: int | None = None,
) -> dict[str, torch.Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).
    """
    assert len(prompt_strs) == len(output_strs)

    # Ensure tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    batch_size = len(prompt_strs)
    input_ids_list = []
    labels_list = []
    mask_list = []

    # Tokenize all
    p_ids_batch = [tokenizer.encode(p, add_special_tokens=False) for p in prompt_strs]
    o_ids_batch = [tokenizer.encode(o, add_special_tokens=False) for o in output_strs]
    eos_id = tokenizer.eos_token_id

    if target_len is None:
        # Determine max length in batch for padding if we want to return a stacked tensor
        # Actually, if target_len is None, we might be using this for single examples or packing.
        # Let's use the actual length of each and pad to max if we must stack.
        lengths = [len(p) + len(o) + 1 for p, o in zip(p_ids_batch, o_ids_batch)]
        actual_target_len = max(lengths) if lengths else 0
    else:
        actual_target_len = target_len

    for i in range(batch_size):
        p_ids = p_ids_batch[i]
        o_ids = o_ids_batch[i]
        full_ids = p_ids + o_ids + [eos_id]

        # input_ids: full_ids[:actual_target_len]
        # labels: full_ids[1:actual_target_len+1]

        # We need to handle the case where we want the FULL sequence without capping.
        # If target_len is None, we want all tokens.
        if target_len is None:
            # For input_ids, we take everything except the last label?
            # No, if we want [P0...EOS], labels are [P1...EOS, -100]?
            # Let's match the "standard" LM shifting if target_len is None.
            curr_input_ids = torch.tensor(full_ids[:-1], dtype=torch.long)
            curr_labels = torch.tensor(full_ids[1:], dtype=torch.long)

            # Mask for labels [P1...Rm, EOS]
            # O start at len(p_ids) in full_ids. In labels, that's len(p_ids) - 1.
            # O end at len(p_ids) + len(o_ids). In labels, that's len(p_ids) + len(o_ids) - 1.
            curr_mask = torch.zeros_like(curr_labels, dtype=torch.bool)
            start_mask = len(p_ids) - 1
            end_mask = len(p_ids) + len(o_ids) - 1

            start_mask = max(0, start_mask)
            end_mask = max(0, end_mask)
            curr_mask[start_mask:end_mask] = True
        else:
            curr_input_ids = torch.tensor(
                full_ids[:actual_target_len], dtype=torch.long
            )
            curr_labels = torch.tensor(
                full_ids[1 : actual_target_len + 1], dtype=torch.long
            )
            curr_mask = torch.zeros_like(curr_labels, dtype=torch.bool)
            start_mask = len(p_ids) - 1
            end_mask = len(p_ids) + len(o_ids) - 1
            start_mask = max(0, min(start_mask, actual_target_len))
            end_mask = max(0, min(end_mask, actual_target_len))
            curr_mask[start_mask:end_mask] = True

            # Padding
            if len(curr_input_ids) < actual_target_len:
                pad_len = actual_target_len - len(curr_input_ids)
                curr_input_ids = F.pad(
                    curr_input_ids, (0, pad_len), value=tokenizer.pad_token_id
                )
            if len(curr_labels) < actual_target_len:
                pad_len = actual_target_len - len(curr_labels)
                curr_labels = F.pad(
                    curr_labels, (0, pad_len), value=tokenizer.pad_token_id
                )
                curr_mask = F.pad(curr_mask, (0, pad_len), value=False)

        input_ids_list.append(curr_input_ids)
        labels_list.append(curr_labels)
        mask_list.append(curr_mask)

    # If target_len is None, they might be different lengths.
    # But usually this is called with batch_size=1 or we want to stack for a batch.
    # If we want to stack, we must pad to max.
    if target_len is None:
        max_inp = max(len(x) for x in input_ids_list)
        max_lab = max(len(x) for x in labels_list)
        input_ids_list = [
            F.pad(x, (0, max_inp - len(x)), value=tokenizer.pad_token_id)
            for x in input_ids_list
        ]
        labels_list = [F.pad(x, (0, max_lab - len(x)), value=-100) for x in labels_list]
        mask_list = [F.pad(x, (0, max_lab - len(x)), value=False) for x in mask_list]

    input_ids = torch.stack(input_ids_list)
    labels = torch.stack(labels_list)
    response_mask = torch.stack(mask_list)

    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the next-token predictions (i.e., entropy over the vocabulary dimension)."""
    # Fast cast to float for stability and BFloat16 CPU support
    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = torch.exp(log_probs)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """Get the conditional log-probs of the response given the prompt."""
    outputs = model(input_ids)

    logits = outputs.logits.float()

    # logits shape: (batch, seq, vocab)
    log_probs_all = F.log_softmax(logits, dim=-1)

    token_entropy = None
    if return_token_entropy:
        token_entropy = compute_entropy(logits)

    # labels shape: (batch, seq)
    # Handle -100 labels for gather to avoid out-of-bounds error
    labels_for_gather = labels.clone()
    ignore_mask = labels_for_gather == -100
    labels_for_gather[ignore_mask] = 0

    gathered_log_probs = torch.gather(
        log_probs_all, -1, labels_for_gather.unsqueeze(-1)
    ).squeeze(-1)

    # Set log-probs for ignored tokens to 0.0
    gathered_log_probs[ignore_mask] = 0.0

    result = {"log_probs": gathered_log_probs}
    if return_token_entropy:
        result["token_entropy"] = token_entropy

    return result


def masked_mean(
    tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None
) -> torch.Tensor:
    """Compute the mean of the tensor along a dimension, considering only the elements with mask value 1."""
    # tensor * mask -> zero out masked elements
    # Ensure float for mean calculation
    tensor = tensor.float()
    mask = mask.float()
    masked_tensor = tensor * mask

    if dim is None:
        return masked_tensor.sum() / mask.sum()
    else:
        return masked_tensor.sum(dim=dim) / mask.sum(dim=dim)


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant, considering only those elements where mask == 1."""
    tensor = tensor.float()
    mask = mask.float()
    masked_tensor = tensor * mask
    if dim is None:
        return masked_tensor.sum() / normalize_constant
    else:
        return masked_tensor.sum(dim=dim) / normalize_constant
