import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import typer

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

logger = logging.getLogger(__name__)

app = typer.Typer()


def sample_responses(
    model,
    tokenizer,
    prompts: List[str],
    num_samples: int = 4,
    max_new_tokens: int = 512,
    temperature: float = 0.8,
    top_p: float = 0.95,
) -> List[List[str]]:
    """Sample multiple responses for each prompt."""
    device = next(model.parameters()).device

    all_samples = []

    # Process in small batches to avoid OOM
    batch_size = 1  # Simple for now

    for i in tqdm(range(0, len(prompts), batch_size), desc="Sampling"):
        batch_prompts = prompts[i : i + batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_samples,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Decode
        batch_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Structure as List[List[str]] where outer list is per prompt
        for j in range(len(batch_prompts)):
            all_samples.append(batch_outputs[j * num_samples : (j + 1) * num_samples])

    return all_samples


@app.command()
def collect(
    model_path: str = "data/models/Qwen2.5-Math-1.5B",
    data_path: str = "data/MATH/validation.jsonl",
    output_path: str = "expert_trajectories.jsonl",
    num_samples: int = 4,
    limit: int = 10,
    seed: int = 42,
):
    logging.basicConfig(level=logging.INFO)
    logger.info(f"Loading model from {model_path}...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
    )
    model.eval()

    # Load data
    with open("cs336_alignment/prompts/r1_zero.prompt", "r") as f:
        prompt_template = f.read()

    prompts = []
    ground_truths = []
    original_examples = []

    with open(data_path, "r") as f:
        for line in f:
            ex = json.loads(line)
            question = ex.get("problem") or ex.get("question")
            answer = ex.get("solution") or ex.get("answer")
            prompts.append(prompt_template.format(question=question))
            ground_truths.append(answer)
            original_examples.append(ex)
            if limit > 0 and len(prompts) >= limit:
                break

    # Sample
    samples_per_prompt = sample_responses(
        model, tokenizer, prompts, num_samples=num_samples
    )

    # Filter
    correct_trajectories = []
    for prompt, samples, gt, orig in zip(
        prompts, samples_per_prompt, ground_truths, original_examples
    ):
        for sample in samples:
            # Judge
            # Note: sample in transformers includes the prompt. We need to extract the assistant response.
            # R1_ZERO_PROMPT_TEMPLATE ends with "Assistant: <think>"
            # But tokenizer.decode(outputs, skip_special_tokens=True) might have different formatting.
            # Let's use a more robust split.

            # The prompt ends with `<think>`
            # We want the text AFTER the prompt.
            # But model.generate returns the whole thing.

            # Find the position where the prompt ends
            # Actually, it's easier to just pass the whole thing to the reward function
            # if the reward function handles it.
            # Let's check r1_zero_reward_fn.

            result = r1_zero_reward_fn(sample, gt)
            if result["reward"] > 0:
                # Found a correct trajectory!
                # We save it in a format suitable for SFT.
                # SFT script expects 'prompt' and 'output'.
                # We need to extract the response part from the sample.

                # Split by "<think>"
                parts = sample.split("<think>")
                if len(parts) > 1:
                    # assistant_response = "<think>" + everything after first <think>
                    # But prompt already had "<think>".
                    # Actually, if we use PROMPT_TEMPLATE.format(question=question),
                    # the prompt is "... Assistant: <think>"
                    # So the sample starts with the prompt.

                    # Let's be precise:
                    prompt_text = prompt
                    # remove special tokens if any that skip_special_tokens missed
                    assistant_response = sample[len(prompt_text) :].strip()
                    # Re-add "<think>" if it was stripped or part of prompt
                    # The prompt ends with "<think>". So assistant_response starts AFTER that.
                    # We want the trajectory to be (prompt_without_think, think+rest)

                    # Prompt in sft.py is:
                    # "User: {instruction}\nAssistant: <think>"
                    # So tokenize_prompt_and_output will use that.

                    # Let's save the cleaned instruction and the response.
                    # R1_ZERO_PROMPT_TEMPLATE has:
                    # "User: {instruction}\nAssistant: <think>"
                    # Instructions: everything between "User: " and "\nAssistant: <think>"

                    # Actually, let's just save 'prompt' and 'output' as expected by SFT.
                    # But omit the "<think>" from prompt so SFT can re-add it?
                    # SFT's PROMPT_TEMPLATE has it.

                    instruction = orig.get("problem") or orig.get("question")
                    correct_trajectories.append(
                        {
                            "prompt": instruction,
                            "output": "<think>" + assistant_response,
                        }
                    )

    # Save
    with open(output_path, "w") as f:
        for traj in correct_trajectories:
            f.write(json.dumps(traj) + "\n")

    logger.info(
        f"Collected {len(correct_trajectories)} correct trajectories and saved to {output_path}"
    )


if __name__ == "__main__":
    app()
