import json
import torch
import typer
from typing import Callable, List
from transformers import AutoModelForCausalLM, AutoTokenizer, Pipeline, pipeline
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from tqdm import tqdm

app = typer.Typer()


def evaluate_transformers(
    model_name: str,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    ground_truths: List[str],
    output_path: str,
    max_new_tokens: int = 1024,
) -> None:
    print(f"Loading model and tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use bfloat16 if supported, else float32 for CPU
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map="cpu"
    )
    model.eval()

    print(f"Generating responses for {len(prompts)} prompts using transformers...")
    results = []
    total_reward = 0.0
    total_format_reward = 0.0
    total_answer_reward = 0.0

    for i in tqdm(range(len(prompts))):
        prompt = prompts[i]
        ground_truth = ground_truths[i]

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedier for baseline
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

        # As per PDF, we might want to include the stop string if the model stopped there.
        # But skip_special_tokens=True usually removes it.
        # Fixed reward function expects tags.

        scores = reward_fn(generated_text, ground_truth)

        total_reward += scores["reward"]
        total_format_reward += scores.get("format_reward", 0.0)
        total_answer_reward += scores.get("answer_reward", 0.0)

        results.append(
            {
                "prompt": prompt,
                "generated_text": generated_text,
                "ground_truth": ground_truth,
                "scores": scores,
            }
        )

    accuracy = total_reward / len(prompts)
    format_accuracy = total_format_reward / len(prompts)
    answer_accuracy = total_answer_reward / len(prompts)

    print(f"Evaluation Complete.")
    print(f"Average Reward (Accuracy): {accuracy:.4f}")
    print(f"Format Reward: {format_accuracy:.4f}")
    print(f"Answer Reward: {answer_accuracy:.4f}")

    with open(output_path, "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")

    summary_path = output_path.replace(".jsonl", "_metrics.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "accuracy": accuracy,
                "format_accuracy": format_accuracy,
                "answer_accuracy": answer_accuracy,
                "total_examples": len(prompts),
            },
            f,
            indent=2,
        )

    print(f"Results saved to {output_path}")


@app.command()
def main(
    model_path: str = "data/models/Qwen2.5-Math-1.5B",
    data_path: str = "data/MATH/validation.jsonl",
    output_path: str = "zero_shot_results_transformers.jsonl",
    limit: int = 5,
):
    # Load data
    print(f"Loading data from {data_path}...")
    prompts = []
    ground_truths = []

    with open("cs336_alignment/prompts/r1_zero.prompt", "r") as f:
        prompt_template = f.read()

    with open(data_path, "r") as f:
        for line in f:
            example = json.loads(line)
            question = example.get("problem") or example.get("question")
            answer = example.get("solution") or example.get("answer")

            formatted_prompt = prompt_template.format(question=question)
            prompts.append(formatted_prompt)
            ground_truths.append(answer)

            if limit > 0 and len(prompts) >= limit:
                break

    evaluate_transformers(
        model_name=model_path,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        ground_truths=ground_truths,
        output_path=output_path,
    )


if __name__ == "__main__":
    app()
